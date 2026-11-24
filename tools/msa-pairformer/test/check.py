#!/usr/bin/env -S python3 -I
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd

out = Path(__import__('sys').argv[1])
metrics_path = out / 'metrics.jsonl'
row = json.loads(metrics_path.read_text().strip())
assert row['depth'] >= 2, row
assert row['coverage_rows'] < 41, row
assert row['query_homolog_max_identity'] <= .99, row
assert row['neff'] > 1 and row['neff_sqrt_l'] > 0, row
assert row['ncount'] == 120, row
assert np.isfinite(row['nll_sum']) and np.isfinite(row['entropy_sum']), row
assert row['entropy_sum'] > 0 and row['inference_seconds'] > 0, row
meta = json.loads((out / 'fixture' / 'mask.json').read_text())
assert meta['query_row'] == 0 and meta['homolog_rows_masked'] is False
assert len(meta['logit_columns']) == 20 and meta['k'] == 7
assert max(meta['fold_sizes']) - min(meta['fold_sizes']) <= 1
assert meta['weights_revision'] == '7563e77a87536b5572f91683c39073ef348639a4'
imports = pd.read_parquet(out / 'fixture' / 'logits.parquet')
assert len(imports) == 120 and imports['has_conditional_logits'].all()
assert imports['logits_f16'].map(len).eq(40).all()
logits = np.stack([np.frombuffer(v, dtype='<f2') for v in imports['logits_f16']])
assert logits.shape == (120, 20) and np.isfinite(logits).all()
assert (imports.groupby('fold').size() >= 17).all(), imports.groupby('fold').size()
assert sorted(set(imports['fold'])) == list(range(7))
assert imports.groupby('target_index').size().eq(6).all()
references = np.load(out / 'fixture' / 'contacts.npz')['contacts']
# MSA Pairformer contacts compare virtual Cbeta coordinates, not coordinates
# from this synthetic input. Structural precision is measured on real PDB data.
assert references.shape == (120, 120) and np.isfinite(references).all()
assert (references >= 0).all() and (references <= 1).all()
assert np.allclose(references.astype(np.float32), references.T.astype(np.float32), atol=.002)
assert len(list(references.flat)) == 14400
# Re-running a completed chain exercises resumability, not duplicate metrics.
assert len(metrics_path.read_text().splitlines()) == 1
print('PASS: model loaded; query-only 7-fold logits are (120, 20) float16; contacts and filtering metadata valid')
print('Test summary:', {k: row[k] for k in ['depth', 'neff', 'preprocess_seconds', 'inference_seconds', 'peak_gpu_mib']})
