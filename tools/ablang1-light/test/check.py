"""Check AbLang1's command outputs against its upstream package inference paths."""
import math
import os
import pathlib
import sys

import ablang
import numpy as np
import torch

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
UPSTREAM_ORDER = "MRHKDESTNQCGPAVIFYWL"


def reference_matrix(model, ids, positions):
    # Keep upstream's native 20-column order. Alphabetical matrix columns are
    # selected separately below, using actual vocabulary ids rather than an
    # assumed position-to-residue mapping.
    rows = []
    with torch.inference_mode():
        for position in positions:
            batch = ids.clone()
            batch[0, position] = model.tokenizer.vocab_to_token["*"]
            rows.append(model.AbLang(batch)[0, position, 1:21].float().cpu().numpy())
    return np.stack(rows)


out = pathlib.Path(sys.argv[1])
root = pathlib.Path(__file__).parent
sequence = (root / "sequences.fasta").read_text().splitlines()[1]
masked_sequence = (root / "masked.fasta").read_text().splitlines()[1]
assert len(sequence) == 109
assert masked_sequence.count("<mask>") == 1
assert len(masked_sequence.replace("<mask>", "N")) == len(sequence)
chain = os.environ["ABLANG1_CHAIN"]
model = ablang.pretrained(chain=chain, model_folder=os.environ["ABLANG1_MODEL"],
                          random_init=False, ncpu=1, device="cuda")
model.freeze()
original_ids = model.tokenizer([sequence], pad=True, device="cuda")
positions = torch.arange(1, len(sequence) + 1, device="cuda")
reference = reference_matrix(model, original_ids, positions)
reference = torch.from_numpy(reference).log_softmax(dim=-1).numpy()
# The reference keeps columns in the tokenizer's native residue order. An
# independent numeric check cannot reuse RESIDUES.index over an implicit 1:21
# slice: that was the mapping error this test is meant to catch.
# Verify the claim this reference relies on without reusing it as a lookup:
# the native columns are the model tokenizer's residue ids minus one, and a
# distinct hard-coded upstream order must agree with that independent mapping.
assert {model.tokenizer.vocab_to_token[aa] - 1: aa for aa in RESIDUES} == dict(enumerate(UPSTREAM_ORDER))
true_columns = np.array([model.tokenizer.vocab_to_token[residue] - 1 for residue in sequence])
expected = float(reference[np.arange(len(sequence)), true_columns].sum())
rows = [line.rstrip("\n").split("\t") for line in (out / "pll.tsv").open()]
assert rows[0] == ["id", "pll"] and len(rows) == 2 and rows[1][0] == "chain1", rows
pll = float(rows[1][1])
assert math.isfinite(pll) and abs(pll - expected) < 0.01, (pll, expected)
assert pll / len(sequence) > math.log(1 / 20), (pll, len(sequence), math.log(1 / 20))
print(f"chain={chain}; pll={pll:.6f}; upstream tokenizer-indexed masked-logit sum={expected:.6f}")
print(f"mean_per_residue_pll={pll / len(sequence):.6f}; uniform_log_probability={math.log(1 / 20):.6f}")

with torch.inference_mode():
    hidden = model.AbRep(original_ids).last_hidden_states[0, 1:len(sequence) + 1]
    upstream_embedding = hidden.mean(0).cpu().numpy()
embeddings = np.load(out / "emb.npz")
assert list(embeddings["ids"]) == ["chain1"]
assert embeddings["embeddings"].shape == (1, 768) and embeddings["embeddings"].dtype == np.float32
print("embedding_max_difference", np.max(np.abs(embeddings["embeddings"][0] - upstream_embedding)))
assert np.allclose(embeddings["embeddings"][0], upstream_embedding, atol=1e-4)

# Call the package's own public restore mode with its native '*' mask token.
expected_fasta = [">chain1", model([masked_sequence.replace("<mask>", "*")], mode="restore")[0]]
assert (out / "filled.fasta").read_text().splitlines() == expected_fasta
assert sum(a != b for a, b in zip(expected_fasta[1], sequence)) <= 1
print("fill exactly matches upstream ablang.pretrained(..., mode='restore')", expected_fasta[1])

saved = np.load(out / "matrix.npz")
assert list(saved["ids"]) == ["chain1"]
assert "".join(saved["amino_acids"]) == RESIDUES
actual = saved["chain1"]
assert actual.shape == (len(sequence), 20) and actual.dtype == np.float32
for alphabet_column, residue in enumerate(RESIDUES):
    upstream_column = model.tokenizer.vocab_to_token[residue] - 1
    assert 1 <= upstream_column + 1 <= 20
    difference = float(np.max(np.abs(actual[:, alphabet_column] - reference[:, upstream_column])))
    assert difference < 2e-4, (residue, alphabet_column, upstream_column, difference)
assert np.allclose(np.exp(actual).sum(axis=1), 1, atol=2e-5)
columns = np.array([RESIDUES.index(residue) for residue in sequence])
assert abs(actual[np.arange(len(sequence)), columns].sum() - pll) < 0.001
assert abs(actual[np.arange(len(sequence)), columns].sum() - expected) < 0.01
print("matrix", actual.shape, actual.dtype, "matches upstream tokenizer-indexed masked logits column by column")
print("ok")
