"""Check all four commands against upstream's unpaired BERT masked-LM calls."""
import math
import os
import pathlib
import sys

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def read(path):
    return path.read_text().splitlines()[1]


def encode(tokenizer, sequence):
    """Follow upstream Appendix A's single-chain spaced tokenization."""
    text = " ".join(sequence.replace("<mask>", "X"))
    tokens = tokenizer.tokenize(text)
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to("cuda")
    positions = [index + 1 for index, token in enumerate(tokens) if token in RESIDUES or token == "X"]
    assert len(positions) == len(sequence.replace("<mask>", "M"))
    assert ids.shape[1] == len(positions) + 2
    masked_positions = [position for position in positions if tokens[position - 1] == "X"]
    ids[0, masked_positions] = tokenizer.mask_token_id
    return ids, torch.tensor(positions, device="cuda"), text


def matrix(model, tokenizer, ids, positions, allowed):
    rows = []
    with torch.inference_mode():
        for position in positions:
            batch = ids.clone()
            batch[0, position] = tokenizer.mask_token_id
            logits = model(input_ids=batch).logits[0, position].log_softmax(-1)
            rows.append(logits[allowed].float().cpu().numpy())
    return np.stack(rows)


out = pathlib.Path(sys.argv[1])
root = pathlib.Path(__file__).parent
sequence = read(root / "sequences.fasta")
masked_sequence = read(root / "masked.fasta")
path = os.environ["IGBERT_UNPAIRED_MODEL"]
tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=False)
model, loading_info = BertForMaskedLM.from_pretrained(path, output_loading_info=True)
assert not loading_info["missing_keys"], loading_info["missing_keys"]
assert not loading_info["unexpected_keys"], loading_info["unexpected_keys"]
assert not loading_info["mismatched_keys"], loading_info["mismatched_keys"]
model = model.eval().cuda()
print("load_keys", {key: loading_info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys")})
ids, positions, single_text = encode(tokenizer, sequence)
print("single_chain_format", single_text[:25], "...", single_text[-25:], "length", ids.shape[1])
allowed = torch.tensor(tokenizer.convert_tokens_to_ids(list(RESIDUES)), device="cuda")
reference = matrix(model, tokenizer, ids, positions, allowed)
columns = [RESIDUES.index(token) for token in tokenizer.convert_ids_to_tokens(ids[0, positions])]
expected = float(reference[np.arange(len(columns)), columns].sum())
rows = [line.rstrip("\n").split("\t") for line in (out / "pll.tsv").open()]
assert rows[0] == ["id", "pll"] and len(rows) == 3, rows
assert [row[0] for row in rows[1:]] == ["heavy1", "light1"]
pll = float(rows[2][1])
assert math.isfinite(pll) and abs(pll - expected) < 0.002, (pll, expected)
print(f"light_chain_pll={pll:.6f}; upstream BERT full-vocabulary masked sum={expected:.6f}")

with torch.inference_mode():
    upstream_embedding = model.bert(input_ids=ids).last_hidden_state[0, positions].mean(0).cpu().numpy()
embeddings = np.load(out / "emb.npz")
assert list(embeddings["ids"]) == ["heavy1", "light1"]
assert embeddings["embeddings"].shape == (2, 1024) and embeddings["embeddings"].dtype == np.float32
print("embedding_max_difference", np.max(np.abs(embeddings["embeddings"][1] - upstream_embedding)))
assert np.allclose(embeddings["embeddings"][1], upstream_embedding, atol=2e-5)

masked_ids, masked_positions, _ = encode(tokenizer, masked_sequence)
with torch.inference_mode():
    logits = model(input_ids=masked_ids).logits[0].log_softmax(-1)
    predictions = logits[:, allowed].argmax(-1).tolist()
original = masked_sequence.replace("<mask>", "M")
filled = "".join(RESIDUES[predictions[position]] if int(masked_ids[0, position]) == tokenizer.mask_token_id
                 else residue for position, residue in zip(masked_positions.tolist(), original))
expected_fasta = [">light1", filled]
assert (out / "filled.fasta").read_text().splitlines() == expected_fasta
assert sum(a != b for a, b in zip(filled, sequence)) <= 2
print("fill exactly matches independent BERT argmax infilling", expected_fasta[1])

saved = np.load(out / "matrix.npz")
assert list(saved["ids"]) == ["heavy1", "light1"]
assert "".join(saved["amino_acids"]) == RESIDUES
actual = saved["light1"]
assert actual.shape == (len(sequence), 20) and actual.dtype == np.float32
assert np.allclose(actual, reference, atol=2e-4), float(np.max(np.abs(actual - reference)))
assert abs(actual[np.arange(len(columns)), columns].sum() - pll) < 0.001
print("matrix", actual.shape, actual.dtype, "matches independent full-vocabulary BERT log-probabilities")
print("ok")
