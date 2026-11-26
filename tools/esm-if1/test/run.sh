#!/usr/bin/env bash
# Actual CLI invocations on a 520-residue complex plus a small single chain.
# Only temporary outputs are written; every process loads baked, verified weights.
set -euo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d /tmp/esm-if1.XXXXXX)
trap 'rm -rf "$T"' EXIT
ESM_IF1_TEST_OUT="$T" python3 -I - <<'PY'
import os
from pathlib import Path
import runpy
api = runpy.run_path("/usr/local/bin/esm-if1", run_name="esm_if1_library")
out = Path(os.environ["ESM_IF1_TEST_OUT"])
for stem in ("antibody-cif", "antibody-pdb", "single-cif", "single-pdb"):
    records = api["fasta"](stem + ".fasta")
    assert [n for n, _ in records] == ["native", "shuffled"]
PY
esm-if1 design --structure antibody.cif --chain H --context-chains L P \
  --out "$T/antibody.cif.fa" --id antibody --num-designs 1 --temperature 0.1 --seed 7
esm-if1 score --structure antibody.cif --chain H --context-chains L,P \
  --sequences antibody-cif.fasta --out "$T/selected-cif.score.tsv" --per-residue "$T/antibody-cif.res.npz"
esm-if1 score --structure antibody.pdb --chain H --context-chains L P \
  --sequences antibody-pdb.fasta --out "$T/selected-pdb.score.tsv"
esm-if1 score --structure antibody.cif --chain H --multichain-backbone \
  --sequences antibody-cif.fasta --out "$T/all-cif.score.tsv"
esm-if1 embed --structure antibody.cif --chain H --context-chains L,P \
  --out "$T/antibody.embed.npz" --id antibody
esm-if1 design --structure single.pdb --out "$T/single.pdb.fa" \
  --num-designs 1 --temperature 0.1 --seed 1
esm-if1 score --structure single.pdb --chain C --sequences single-pdb.fasta \
  --out "$T/single-pdb.score.tsv"
esm-if1 score --structure single.cif --chain C --sequences single-cif.fasta \
  --out "$T/single-cif.score.tsv"
# Original mixed PDB chain A/B/C verifies DNA does not become protein context.
python3 -I - <<'PY'
import runpy
from types import SimpleNamespace
api = runpy.run_path("/usr/local/bin/esm-if1", run_name="esm_if1_library")
assert api["prepare"](SimpleNamespace(structure="mixed.cif", chain="C", context_chains=[], multichain_backbone=False))[1] == api["fasta"]("single-cif.fasta")[0][1]
print("PASS: mixed DNA/protein structure filters to 93-residue protein C")
PY
python3 check.py "$T"
