#!/usr/bin/env -S python3 -I
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

out = Path(sys.argv[1])
k = int(sys.argv[2])
assert k in (2, 7)
row = json.loads((out / 'metrics.jsonl').read_text().strip())
assert row['requested_depth'] == 16 and row['depth'] >= 2, row
assert row['coverage_rows'] < 41, row
assert row['query_homolog_max_identity'] <= .99, row
assert row['neff'] > 1 and row['neff_sqrt_l'] > 0, row
assert row['ncount'] == (36 if k == 2 else 120), row
assert np.isfinite(row['nll_sum']) and np.isfinite(row['entropy_sum']), row
assert row['entropy_sum'] > 0 and row['inference_seconds'] > 0, row
meta = json.loads((out / 'fixture' / 'mask.json').read_text())
assert meta['query_row'] == 0 and meta['homolog_rows_masked'] is False
assert meta['weights_revision'] == '7563e77a87536b5572f91683c39073ef348639a4'
assert meta['logit_columns'] == list('ARNDCEQGHILKMFPSTWYV')
assert meta['k'] == k and meta['seed'] == 42 and meta['requested_depth'] == 16
imports = pd.read_parquet(out / 'fixture' / 'logits.parquet')
assert len(imports) == 120 and imports['logits_f16'].map(len).eq(40).all()
assert imports.groupby('target_index').size().eq(6).all()
conditional = imports['has_conditional_logits'].to_numpy()
assert conditional.sum() == sum(meta['fold_sizes']) == row['ncount']
logits = np.stack([np.frombuffer(v, dtype='<f2') for v in imports['logits_f16']])
assert logits.shape == (120, 20) and np.isfinite(logits[conditional]).all()
references = np.load(out / 'fixture' / 'contacts.npz')['contacts']
assert references.shape == (120, 120) and np.isfinite(references).all()
assert (references >= 0).all() and (references <= 1).all()
assert np.allclose(references.astype(np.float32), references.T.astype(np.float32), atol=.002)
if k == 7:
    assert conditional.all() and meta['fold_sizes'] == [18, 17, 17, 17, 17, 17, 17]
    assert sorted(set(imports['fold'])) == list(range(7))
    assert imports['target_index'].to_numpy().tolist() == [i % 20 for i in range(20)] * 6
else:
    assert conditional.sum() == 35 and meta['fold_sizes'] == [18, 17]
    assert sorted(set(imports.loc[conditional, 'fold'])) == [0, 1]
    assert np.isnan(logits[~conditional]).all()
    full = Path(sys.argv[3])
    other = pd.read_parquet(full / 'fixture' / 'logits.parquet')
    assert imports.loc[conditional, ['position', 'fold', 'target_index']].reset_index(drop=True).equals(
        other.loc[other['has_conditional_logits'] & other['fold'].isin([0, 1]),
                  ['position', 'fold', 'target_index']].reset_index(drop=True))
    other_logits = np.stack([np.frombuffer(v, dtype='<f2') for v in other['logits_f16']])
    assert np.array_equal(logits[conditional], other_logits[conditional])
    assert np.array_equal(references, np.load(full / 'fixture' / 'contacts.npz')['contacts'])
assert len((out / 'metrics.jsonl').read_text().splitlines()) == 1
print(f'PASS: model loaded; query-only K={k} logits are (120, 20) float16; contacts and filtering metadata valid')
print('Test summary:', {key: row[key] for key in ['depth', 'neff', 'preprocess_seconds', 'inference_seconds', 'peak_gpu_mib']})
