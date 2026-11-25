"""Check all four outputs and reproduce upstream's own pseudo_log_likelihood."""
import math
import os
import pathlib
import sys

import ablang2
import numpy as np

out = pathlib.Path(sys.argv[1])
root = pathlib.Path(__file__).parent
heavy = (root / "heavy.fasta").read_text().splitlines()[1]
light = (root / "light.fasta").read_text().splitlines()[1]
masked_heavy = (root / "masked-heavy.fasta").read_text().splitlines()[1]
masked_light = (root / "masked-light.fasta").read_text().splitlines()[1]
model = ablang2.pretrained(model_to_use=os.environ["ABLANG2_MODEL"], random_init=False,
                           ncpu=1, device="cpu")
upstream_mean = float(model([[heavy, light]], mode="pseudo_log_likelihood")[0])
expected = upstream_mean * (len(heavy) + len(light))
assert math.isfinite(upstream_mean), upstream_mean

rows = [line.rstrip("\n").split("\t") for line in (out / "pll.tsv").open()]
assert rows[0] == ["id", "pll"] and len(rows) == 2 and rows[1][0] == "brii1", rows
pll = float(rows[1][1])
assert math.isfinite(pll) and abs(pll - expected) < 0.02, (pll, expected)
print(f"pll={pll:.6f}; upstream mean={upstream_mean:.8f}, mean*residues={expected:.6f}")

emb = np.load(out / "emb.npz")
assert list(emb["ids"]) == ["brii1"]
assert emb["embeddings"].shape == (1, 480) and emb["embeddings"].dtype == np.float32
assert np.isfinite(emb["embeddings"]).all()
matrices = np.load(out / "matrix.npz")
assert list(matrices["ids"]) == ["brii1"]
assert "".join(matrices["amino_acids"].tolist()) == "ACDEFGHIKLMNPQRSTVWY"
matrix = matrices["brii1"]
assert matrix.shape == (len(heavy) + len(light), 20) and matrix.dtype == np.float32
assert np.allclose(np.exp(matrix).sum(axis=1), 1, atol=2e-5)
assert abs(matrix[np.arange(len(matrix)),
                  np.array(["ACDEFGHIKLMNPQRSTVWY".index(aa) for aa in heavy + light])].sum() - pll) < 0.001

filled = (out / "filled.fasta").read_text().splitlines()
assert filled[0] == ">brii1" and len(filled) == 2
assert len(filled[1]) == len(heavy) + len(light) + 1 and "*" not in filled[1]
mask_positions = (masked_heavy.index("<mask>"), len(heavy) + 1 + masked_light.index("*"))
assert "".join(aa for i, aa in enumerate(filled[1]) if i not in mask_positions) == heavy + "|" + light
assert all(filled[1][i] in "ACDEFGHIKLMNPQRSTVWY" for i in mask_positions)
print("matrix", matrix.shape, matrix.dtype, "; embedding (1, 480); paired infilling preserves context")
print("ok")
