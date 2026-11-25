"""Check the outputs against upstream AntiBERTy's own scorer and fill implementation."""
import math
import os
import pathlib
import sys

from antiberty import AntiBERTy, AntiBERTyRunner
from antiberty.tokenizer import MASK_CHAR
import numpy as np
import torch

out = pathlib.Path(sys.argv[1])
root = pathlib.Path(__file__).parent
sequence = (root / "input.fasta").read_text().splitlines()[1]
masked = (root / "masked.fasta").read_text().splitlines()[1].replace("<mask>", MASK_CHAR)
runner = AntiBERTyRunner(device="cuda", checkpoint_path=os.environ["ANTIBERTY_WEIGHTS_DIR"])
upstream_mean = float(runner.pseudo_log_likelihood([sequence], batch_size=64)[0])
expected = upstream_mean * len(sequence)
upstream_fill = runner.fill_masks([masked])
# Verify that no key needed by the AntiBERTy model was silently initialized.
model, loading_info = AntiBERTy.from_pretrained(
    os.environ["ANTIBERTY_WEIGHTS_DIR"], attn_implementation="eager", output_loading_info=True)
missing = sorted(loading_info["missing_keys"])
unexpected = sorted(loading_info["unexpected_keys"])
mismatched = sorted(loading_info["mismatched_keys"])
assert not loading_info["error_msgs"], loading_info["error_msgs"]
assert missing == [], ("randomly initialized missing weights", missing)
print(f"load_keys: missing={missing}, unexpected={unexpected}, mismatched_sizes={mismatched}")
print(f"upstream_expectations: mean_pll={upstream_mean:.8f}, sum_pll={expected:.6f}; fill={upstream_fill[0]}")

rows = [line.rstrip("\n").split("\t") for line in (out / "pll.tsv").open()]
assert rows[0] == ["id", "pll"] and len(rows) == 2 and rows[1][0] == "brii1", rows
pll = float(rows[1][1])
assert math.isfinite(pll) and abs(pll - expected) < 0.002, (pll, expected)
print(f"pll={pll:.6f}; upstream mean*residues={expected:.6f}")

embeddings = np.load(out / "emb.npz")["embeddings"]
upstream_embedding = runner.embed([sequence], hidden_layer=-1)[0][1:-1].mean(0).float().cpu().numpy()
print("embedding_diagnostics:",
      f"tool_range=({embeddings.min():.6f}, {embeddings.max():.6f})",
      f"upstream_range=({upstream_embedding.min():.6f}, {upstream_embedding.max():.6f})",
      f"max_difference={np.max(np.abs(embeddings[0] - upstream_embedding)):.6f}")
assert embeddings.shape == (1, 512) and embeddings.dtype == np.float32
assert np.allclose(embeddings[0], upstream_embedding, atol=2e-5), np.max(np.abs(embeddings[0] - upstream_embedding))
print("embedding matches upstream last-layer residue mean")

matrices = np.load(out / "matrix.npz")
assert list(matrices["ids"]) == ["brii1"]
assert "".join(matrices["amino_acids"].tolist()) == "ACDEFGHIKLMNPQRSTVWY"
matrix = matrices["brii1"]
assert matrix.shape == (len(sequence), 20) and matrix.dtype == np.float32
assert np.allclose(np.exp(matrix).sum(axis=1), 1, atol=2e-5)
assert abs(matrix[np.arange(len(matrix)), ["ACDEFGHIKLMNPQRSTVWY".index(aa) for aa in sequence]].sum() - pll) < 0.001
# Masking its five non-amino-acid logits to -inf leaves no probability outside
# these 20 columns, so standard and full-vocab normalize identically.

filled = (out / "filled.fasta").read_text().splitlines()
assert filled == [">brii1", upstream_fill[0]], (filled, upstream_fill)
assert len(filled[1]) == len(sequence)
assert all(filled[1][i] == sequence[i] for i in range(len(sequence)) if masked[i] != MASK_CHAR)
print("matrix", matrix.shape, matrix.dtype, "; fill exactly matches upstream")
print("ok")
