"""check.py <contacts.npz> <reference.npz>: the contacts map matches oplm's own computation.

The reference was computed with oplm's compute_categorical_jacobian and
categorical_jacobian_to_contact_map (briney/oplm at 704e01f5ed4b8322bb82e0c961b2c16361c47fa2)
on the same DPLM weights in bf16. The tolerances are the skill's: Pearson r >= 0.98
and top-L long-range overlap >= 0.8, allowing GPU kernel noise; symmetry is exact.
"""
import sys

import numpy as np

PEARSON_FLOOR = 0.98   # the skill's tolerance for GPU kernel noise
TOP_L_OVERLAP_FLOOR = 0.8
MIN_SEQ_SEP = 24       # oplm's long-range separation
LENGTHS = {"ubiquitin": 76}  # the fixture's sequences; L for top-L


def top_l_pairs(scores, l):
    """The top-l long-range pairs of an (L, L) map, as a set of (i, j) with i < j, ties by position."""
    n = scores.shape[0]
    iu, ju = np.triu_indices(n, k=MIN_SEQ_SEP)
    order = np.lexsort((ju, iu, -scores[iu, ju].astype(np.float64)))
    return {(int(iu[k]), int(ju[k])) for k in order[:l]}


contacts, reference = (np.load(p) for p in sys.argv[1:3])

assert contacts.files == list(LENGTHS) == reference.files, (contacts.files, reference.files)
for name, l in LENGTHS.items():
    got, want = contacts[name], reference[name].astype(np.float64)
    assert got.shape == (l, l), (name, got.shape)
    assert got.dtype == np.float32, (name, got.dtype)
    assert np.isfinite(got).all(), name
    assert np.array_equal(got, got.T), f"{name}: the map is not exactly symmetric"

    r = float(np.corrcoef(got.ravel(), want.ravel())[0, 1])
    assert r >= PEARSON_FLOOR, f"{name}: Pearson r {r:.4f} < {PEARSON_FLOOR}"

    overlap = len(top_l_pairs(got, l) & top_l_pairs(want, l)) / l
    assert overlap >= TOP_L_OVERLAP_FLOOR, f"{name}: top-{l} long-range overlap {overlap:.2f} < {TOP_L_OVERLAP_FLOOR}"

    print(f"ok: {name} ({l} residues) symmetric, r={r:.4f}, top-{l} overlap={overlap:.2f}")
