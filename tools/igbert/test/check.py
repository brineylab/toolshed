"""Check IgBert against direct Transformers calls using its published paired format."""
import math
import os
import pathlib
import sys

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def read(path):
    return path.read_text().splitlines()[1]


def encode(tokenizer, heavy, light):
    text = " ".join(heavy) + " [SEP] " + " ".join(light)
    tokens = tokenizer(text, return_tensors="pt").to("cuda")
    ids = tokens["input_ids"]
    token_list = tokenizer.convert_ids_to_tokens(ids[0])
    positions = [index for index, token in enumerate(token_list) if token in RESIDUES]
    assert tokenizer.convert_ids_to_tokens(ids[0, :5]) == ["[CLS]", heavy[0], heavy[1], heavy[2], heavy[3]]
    assert len(positions) == len(heavy) + len(light)
    assert ids.shape[1] == len(positions) + 4
    assert tokens["token_type_ids"][0, positions[len(heavy):]].sum() > 0
    return ids, torch.tensor(positions, device="cuda"), tokens["token_type_ids"], text


def matrix(model, tokenizer, ids, positions, token_types):
    rows = []
    with torch.inference_mode():
        for position in positions:
            batch = ids.clone()
            batch[0, position] = tokenizer.mask_token_id
            logits = model(input_ids=batch, token_type_ids=token_types).logits[0, position]
            allowed = torch.tensor(tokenizer.convert_tokens_to_ids(list(RESIDUES)), device="cuda")
            rows.append(logits[allowed].log_softmax(-1).float().cpu().numpy())
    return np.stack(rows)


out = pathlib.Path(sys.argv[1])
root = pathlib.Path(__file__).parent
heavy = read(root / "heavy.fasta")
light = read(root / "light.fasta")
masked_heavy = read(root / "masked-heavy.fasta").replace("<mask>", "[MASK]")
masked_light = read(root / "masked-light.fasta").replace("<mask>", "[MASK]")
path = os.environ["IGBERT_MODEL"]
tokenizer = AutoTokenizer.from_pretrained(path)
model, loading_info = AutoModelForMaskedLM.from_pretrained(path, output_loading_info=True)
assert not loading_info["missing_keys"], loading_info["missing_keys"]
assert not loading_info["mismatched_keys"], loading_info["mismatched_keys"]
model = model.eval().cuda()
print("load_keys", {key: loading_info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys")})
ids, positions, token_types, paired_text = encode(tokenizer, heavy, light)
print("paired_format", paired_text[:25], "...", paired_text[-25:], "length", ids.shape[1])
reference = matrix(model, tokenizer, ids, positions, token_types)
columns = [RESIDUES.index(token) for token in tokenizer.convert_ids_to_tokens(ids[0, positions])]
expected = float(reference[np.arange(len(columns)), columns].sum())
rows = [line.rstrip("\n").split("\t") for line in (out / "pll.tsv").open()]
assert rows[0] == ["id", "pll"] and len(rows) == 2 and rows[1][0] == "brii1", rows
pll = float(rows[1][1])
assert math.isfinite(pll) and abs(pll - expected) < 0.002, (pll, expected)
print(f"pll={pll:.6f}; independently masked IgBert sum={expected:.6f}")

with torch.inference_mode():
    upstream_embedding = model.bert(input_ids=ids, token_type_ids=token_types).last_hidden_state[0, positions].mean(0).cpu().numpy()
embeddings = np.load(out / "emb.npz")
assert list(embeddings["ids"]) == ["brii1"]
assert embeddings["embeddings"].shape == (1, 1024) and embeddings["embeddings"].dtype == np.float32
print("embedding_max_difference", np.max(np.abs(embeddings["embeddings"][0] - upstream_embedding)))
assert np.allclose(embeddings["embeddings"][0], upstream_embedding, atol=2e-5)

masked_ids, masked_positions, masked_types, _ = encode(tokenizer, masked_heavy, masked_light)
with torch.inference_mode():
    logits = model(input_ids=masked_ids, token_type_ids=masked_types).logits[0]
    allowed = torch.tensor(tokenizer.convert_tokens_to_ids(list(RESIDUES)), device="cuda")
    predictions = logits[:, allowed].argmax(-1).tolist()
filled = "".join(RESIDUES[predictions[position]] if int(masked_ids[0, position]) == tokenizer.mask_token_id
                 else token for position, token in zip(masked_positions.tolist(), list(heavy + light)))
expected_fasta = [">brii1", filled[:len(heavy)] + "|" + filled[len(heavy):]]
assert (out / "filled.fasta").read_text().splitlines() == expected_fasta
print("fill exactly matches an independent IgBert infill", expected_fasta[1])

saved = np.load(out / "matrix.npz")
assert list(saved["ids"]) == ["brii1"]
assert "".join(saved["amino_acids"]) == RESIDUES
actual = saved["brii1"]
assert actual.shape == (len(heavy) + len(light), 20) and actual.dtype == np.float32
assert np.allclose(np.exp(actual).sum(1), 1, atol=2e-5)
assert np.allclose(actual, reference, atol=2e-4), float(np.max(np.abs(actual - reference)))
assert abs(actual[np.arange(len(columns)), columns].sum() - pll) < 0.001
print("matrix", actual.shape, actual.dtype, "matches independent per-position IgBert reference")
print("ok")
