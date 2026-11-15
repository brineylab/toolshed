"""check.py <pll.tsv> <emb.npz> <filled.fasta> <contacts.npz>: the outputs of test/run.sh have the
shapes the skill documents, from trained weights, and the contact map matches the oplm reference."""
import math, sys

import numpy as np

# Ubiquitin's pll per residue: trained ESM-1b -0.03 and ESM-2 650M -0.58; the same models untrained -4.3 and -3.5.
PLL_FLOOR = -1.5
IDS = ["peptide", "reversed", "ubiquitin"]
CONTACT_IDS = ["ubiquitin"]  # contacts.fasta, in file order
MIN_SEQ_SEP = 24             # oplm's long-range minimum sequence separation
PEARSON_FLOOR = 0.98         # the tolerance the skill states
TOP_L_FLOOR = 0.8            # the family's measured floors sit well above it

pll_tsv, emb_npz, filled, contacts_npz = sys.argv[1:]

rows = [line.rstrip("\n").split("\t") for line in open(pll_tsv)]
assert rows[0] == ["id", "pll"], rows[0]
assert [r[0] for r in rows[1:]] == IDS, rows
for _, v in rows[1:]:
    assert math.isfinite(float(v)) and float(v) < 0, v
per_residue = float(rows[3][1]) / 76
assert per_residue > PLL_FLOOR, f"ubiquitin pll per residue {per_residue:.2f} is below {PLL_FLOOR}: the weights look untrained"

emb = np.load(emb_npz)
assert emb["embeddings"].shape == (3, 1280) and emb["embeddings"].dtype == np.float32, emb["embeddings"].shape
assert list(emb["ids"]) == IDS, emb["ids"]

seq = open(filled).read().split("\n")[1]
assert "<mask>" not in seq and len(seq) == len("MKTAYIAK.RQISF.KSH"), seq
# the context around the two masks; the model's own calls at them vary with the kernel
assert seq[:8] + seq[9:14] + seq[15:] == "MKTAYIAKRQISFKSH", seq

# The contact map against the reference computed once with oplm's compute_categorical_jacobian
# and categorical_jacobian_to_contact_map (commit 704e01f) on the same weights in bf16.
maps = np.load(contacts_npz)
reference = np.load(__file__.rsplit("/", 1)[0] + "/contacts-reference.npz")
assert list(maps.files) == CONTACT_IDS, maps.files
for name in CONTACT_IDS:
    got, ref = maps[name], reference[name].astype(np.float64)
    L = got.shape[0]
    assert got.dtype == np.float32 and got.shape == (L, L), (name, got.shape, got.dtype)
    assert np.array_equal(got, got.T), f"{name}: not exactly symmetric"
    pearson = float(np.corrcoef(got.ravel().astype(np.float64), ref.ravel())[0, 1])
    assert pearson >= PEARSON_FLOOR, f"{name}: Pearson r {pearson:.4f} < {PEARSON_FLOOR} against the oplm reference"
    i, j = np.triu_indices(L, k=MIN_SEQ_SEP)
    top = lambda a: {(int(i[k]), int(j[k])) for k in np.argsort(-a[i, j])[:L]}
    overlap = len(top(got.astype(np.float64)) & top(ref)) / L
    assert overlap >= TOP_L_FLOOR, f"{name}: top-{L} long-range overlap {overlap:.2f} < {TOP_L_FLOOR}"
    print(f"contacts {name}: L={L} pearson_r={pearson:.4f} top-{L} long-range overlap {overlap:.2f}")
print("ok:", ", ".join(f"{r[0]} pll {r[1]}" for r in rows[1:]), f"| ubiquitin {per_residue:.2f}/residue | filled", seq)
