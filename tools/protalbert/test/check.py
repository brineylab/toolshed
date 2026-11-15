"""check.py <contacts.npz> <ref.npz>: the contacts command's maps match the committed reference,
which was computed with oplm's compute_categorical_jacobian and categorical_jacobian_to_contact_map
(commit 704e01f5ed4b8322bb82e0c961b2c16361c47fa2) on the same weights in bf16."""
import sys

import numpy as np

MIN_SEQ_SEP = 24  # long-range pairs, oplm's default
PEARSON = 0.98    # the skill's tolerance for GPU kernel noise
TOP_L_OVERLAP = 0.8

got_path, ref_path = sys.argv[1:3]
got, ref = np.load(got_path), np.load(ref_path)
ids = sorted(ref.files)
assert sorted(got.files) == ids, (sorted(got.files), ids)
for name in ids:
    m, r = got[name], ref[name].astype(np.float64)
    L = r.shape[0]
    assert m.dtype == np.float32 and m.shape == (L, L), (name, m.dtype, m.shape)
    assert np.isfinite(m).all(), f"{name}: non-finite scores"
    assert np.array_equal(m, m.T), f"{name}: the map is not exactly symmetric"
    pearson = float(np.corrcoef(m.ravel().astype(np.float64), r.ravel())[0, 1])
    assert pearson >= PEARSON, f"{name}: Pearson r {pearson:.4f} is below {PEARSON}"
    iu = np.triu_indices(L, k=MIN_SEQ_SEP)
    top = min(L, len(iu[0]))
    top_m = set(np.argsort(m[iu])[-top:].tolist())
    top_r = set(np.argsort(r[iu])[-top:].tolist())
    overlap = len(top_m & top_r) / top
    assert overlap >= TOP_L_OVERLAP, f"{name}: top-L overlap {overlap:.3f} is below {TOP_L_OVERLAP}"
    print(f"ok: {name} L={L} pearson={pearson:.4f} top-L overlap={overlap:.3f}")
