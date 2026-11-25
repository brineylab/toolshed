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
    rows = []
    with torch.inference_mode():
        for position in positions:
            batch = ids.clone()
            batch[0, position] = model.tokenizer.vocab_to_token["*"]
            # AbLang's likelihood mode returns logits 1:21 directly; retain
            # those same 20 positions and normalize only for log-probabilities.
            rows.append(model.AbLang(batch)[0, position, 1:21].log_softmax(-1).float().cpu().numpy())
    return np.stack(rows)


out = pathlib.Path(sys.argv[1])
root = pathlib.Path(__file__).parent
sequence = (root / "sequences.fasta").read_text().splitlines()[1]
masked_sequence = (root / "masked.fasta").read_text().splitlines()[1]
chain = os.environ["ABLANG1_CHAIN"]
model = ablang.pretrained(chain=chain, model_folder=os.environ["ABLANG1_MODEL"],
                          random_init=False, ncpu=1, device="cuda")
model.freeze()
original_ids = model.tokenizer(sequence, pad=True, device="cuda").unsqueeze(0)
positions = torch.arange(1, len(sequence) + 1, device="cuda")
reference = reference_matrix(model, original_ids, positions)
columns = np.array([RESIDUES.index(residue) for residue in sequence])
expected = float(reference[np.arange(len(sequence)), columns].sum())
rows = [line.rstrip("\n").split("\t") for line in (out / "pll.tsv").open()]
assert rows[0] == ["id", "pll"] and len(rows) == 2 and rows[1][0] == "chain1", rows
pll = float(rows[1][1])
assert math.isfinite(pll) and abs(pll - expected) < 0.01, (pll, expected)
print(f"chain={chain}; pll={pll:.6f}; upstream per-position masked-logit sum={expected:.6f}")

with torch.inference_mode():
    hidden = model.AbRep(original_ids).last_hidden_states[0, 1:len(sequence) + 1]
    upstream_embedding = hidden.mean(0).cpu().numpy()
embeddings = np.load(out / "emb.npz")
assert list(embeddings["ids"]) == ["chain1"]
assert embeddings["embeddings"].shape == (1, 768) and embeddings["embeddings"].dtype == np.float32
print("embedding_max_difference", np.max(np.abs(embeddings["embeddings"][0] - upstream_embedding)))
assert np.allclose(embeddings["embeddings"][0], upstream_embedding, atol=1e-4)

# Call the package's own public restore mode, not merely its logits.
expected_fasta = [">chain1", model([masked_sequence], mode="restore")[0]]
assert (out / "filled.fasta").read_text().splitlines() == expected_fasta
assert sum(a != b for a, b in zip(expected_fasta[1], sequence)) <= 1
print("fill exactly matches upstream ablang.pretrained(..., mode='restore')", expected_fasta[1])

saved = np.load(out / "matrix.npz")
assert list(saved["ids"]) == ["chain1"]
assert "".join(saved["amino_acids"]) == RESIDUES
actual = saved["chain1"]
assert actual.shape == (len(sequence), 20) and actual.dtype == np.float32
assert np.allclose(actual, reference, atol=2e-4), float(np.max(np.abs(actual - reference)))
assert np.allclose(np.exp(actual).sum(axis=1), 1, atol=2e-5)
assert abs(actual[np.arange(len(sequence)), columns].sum() - pll) < 0.001
print("matrix", actual.shape, actual.dtype, "matches upstream independent masked logits")
print("ok")
