"""check.py <contacts.npz>: the contacts command's maps match the committed reference.

The reference (reference.npz) was computed in a campaign run with oplm's own
compute_categorical_jacobian and categorical_jacobian_to_contact_map
(briney/oplm at commit 704e01f5ed4b8322bb82e0c961b2c16361c47fa2) on the same
weights in bf16; weights.sha256 pins the safetensors the image must carry.
Passes iff every map is exactly symmetric, Pearson r >= 0.98 against the
reference, and the top-L long-range (|i - j| >= 24, oplm's definition) pairs
overlap the reference's by >= 0.8.
"""
import hashlib
import os
import sys

import numpy as np

MIN_SEQ_SEP = 24  # oplm's long-range definition (build_structure_pair_score_data)
PEARSON_FLOOR = 0.98
TOPL_OVERLAP_FLOOR = 0.8

contacts_npz = sys.argv[1]

expected = open("weights.sha256").read().strip()
h = hashlib.sha256()
with open(os.path.join(os.environ["E1_MODEL"], "model.safetensors"), "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        h.update(chunk)
assert h.hexdigest() == expected, "weights sha256 mismatch: the image is not the pinned revision"


def top_l_pairs(scores, min_sep=MIN_SEQ_SEP):
    """The top-L scoring long-range pairs, as oplm's precision@L ranks them."""
    n = scores.shape[0]
    i, j = np.triu_indices(n, k=min_sep)
    vals = scores[i, j]
    k = min(n, vals.size)
    top = np.argpartition(vals, -k)[-k:]
    return {(int(i[t]), int(j[t])) for t in top}, k


ref = np.load("reference.npz")
got = np.load(contacts_npz)
assert sorted(got.files) == sorted(ref.files), (sorted(got.files), sorted(ref.files))

for name in sorted(ref.files):
    m, r = got[name], ref[name].astype(np.float64)
    L = r.shape[0]
    assert m.shape == (L, L), (name, m.shape)
    assert m.dtype == np.float32, (name, m.dtype)
    assert np.array_equal(m, m.T), f"{name}: the map is not exactly symmetric"
    pearson = float(np.corrcoef(m.astype(np.float64).ravel(), r.ravel())[0, 1])
    assert pearson >= PEARSON_FLOOR, f"{name}: Pearson r {pearson:.4f} < {PEARSON_FLOOR}"
    mine, k = top_l_pairs(m.astype(np.float64))
    theirs, _ = top_l_pairs(r)
    overlap = len(mine & theirs) / k
    assert overlap >= TOPL_OVERLAP_FLOOR, f"{name}: top-L long-range overlap {overlap:.3f} < {TOPL_OVERLAP_FLOOR}"
    print(f"{name}: L={L} Pearson r {pearson:.4f}, top-L long-range overlap {overlap:.3f}")

print("ok: maps match the oplm reference within the skill's tolerances")
