#!/usr/bin/env -S python3 -I
import ast
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

out = Path(sys.argv[1])
k = int(sys.argv[2])
# The optional fourth argument names the input stem for sharded-output checks;
# the third remains the K=7 reference directory used by the K=2 comparison.
stem = sys.argv[4] if len(sys.argv) == 5 else 'fixture'
assert k in (2, 7)
row = json.loads((out / 'metrics.jsonl').read_text().strip())
assert len((out / 'metrics.jsonl').read_text().splitlines()) == 1
assert row['id'] == stem, row
assert row['requested_depth'] == 16 and row['depth'] >= 2, row
assert row['coverage_rows'] < 41, row
assert row['query_homolog_max_identity'] <= .99, row
assert row['neff'] > 1 and row['neff_sqrt_l'] > 0, row
# Test the insertion regression independently of hhfilter's diversity choices.
program = Path(__file__).parents[1] / 'bin' / 'msa-pairformer'
tree = ast.parse(program.read_text())
helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'project_a3m_row')
namespace = {}
exec(compile(ast.Module(body=[helper], type_ignores=[]), program, 'exec'), namespace)
assert namespace['project_a3m_row']('ACde.F-G', 5) == 'ACF-G'
records = (Path(__file__).parent / 'fixture.a3m').read_text().split('>')[1:]
rows = [''.join(record.splitlines()[1:]) for record in records]
assert any(char.islower() or char == '.' for char in rows[33])
assert all(len(namespace['project_a3m_row'](row, 120)) == 120 for row in rows)
assert row['ncount'] == (35 if k == 2 else 120), row
assert np.isfinite(row['nll_sum']) and np.isfinite(row['entropy_sum']), row
assert row['entropy_sum'] > 0 and row['inference_seconds'] > 0, row
meta = json.loads((out / stem / 'mask.json').read_text())
assert meta['query_row'] == 0 and meta['homolog_rows_masked'] is False
assert meta['weights_revision'] == '7563e77a87536b5572f91683c39073ef348639a4'
assert meta['logit_columns'] == list('ARNDCEQGHILKMFPSTWYV')
assert meta['k'] == k and meta['seed'] == 42 and meta['requested_depth'] == 16
imports = pd.read_parquet(out / stem / 'logits.parquet')
assert len(imports) == 120 and imports['logits_f16'].map(len).eq(40).all()
assert imports.groupby('target_index').size().eq(6).all()
conditional = imports['has_conditional_logits'].to_numpy()
assert conditional.sum() == sum(meta['fold_sizes']) == row['ncount']
logits = np.stack([np.frombuffer(v, dtype='<f2') for v in imports['logits_f16']])
assert logits.shape == (120, 20) and np.isfinite(logits[conditional]).all()
references = np.load(out / stem / 'contacts.npz')['contacts']
assert references.shape == (120, 120) and np.isfinite(references).all()
assert (references >= 0).all() and (references <= 1).all()
assert np.allclose(references.astype(np.float32), references.T.astype(np.float32), atol=.002)
if k == 7:
    assert conditional.all() and meta['fold_sizes'] == [18, 17, 17, 17, 17, 17, 17]
    assert sorted(set(imports['fold'])) == list(range(7))
    query = next(iter((Path(__file__).parent / 'fixture.a3m').read_text().split('>')[1:]))
    expected_query = ['ARNDCEQGHILKMFPSTWYV'.index(aa) for aa in ''.join(query.splitlines()[1:])]
    assert imports['target_index'].to_numpy().tolist() == expected_query
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
if len(sys.argv) == 5:
    reference = Path(sys.argv[3]) / 'fixture'
    unsharded = pd.read_parquet(reference / 'logits.parquet')
    assert imports.drop(columns='id').equals(unsharded.drop(columns='id'))
    assert np.array_equal(references, np.load(reference / 'contacts.npz')['contacts'])
print(f'PASS: model loaded; query-only K={k} logits are (120, 20) float16; contacts and filtering metadata valid')
print('Test summary:', {key: row[key] for key in ['depth', 'neff', 'preprocess_seconds', 'inference_seconds', 'peak_gpu_mib']})
