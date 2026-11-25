#!/usr/bin/env bash
# ssrmsd's test: exercises version reporting, Pyxis python3 -I isolation from a
# read-only working directory, multi-worker batch --dir across AF3 (with decoy
# top-level duplicate) and flat esmfold2 *_sample_*.cif (+ sidecars) roots,
# single-ensemble score on a .tar.gz archive + poly-G-linked .pdb, and manifest
# batch JSONL with a homomer error row; checks all metrics against analytical
# expectations within 1e-4.
set -euo pipefail

FIXTURE=/toolshed/test/fixture
work=$(mktemp -d /tmp/ssrmsd-test.XXXXXX)
trap 'rm -rf "$work"' EXIT

# 1. Version check
ssrmsd --version | grep -Fx 'ssrmsd 0.1.1'

# 2. Pyxis isolation: poisoned PYTHONPATH and minimal distro PATH from read-only cwd
mkdir -p "$work/poison"
printf 'raise RuntimeError("caller PYTHONPATH leaked into container")\n' > "$work/poison/ssrmsd.py"
(
  cd /toolshed/test
  env -i PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$work/poison" ssrmsd --version
) | grep -Fx 'ssrmsd 0.1.1'

# 3. Threshold-crossing batch --dir (3 ensembles, --workers 2 -> ProcessPoolExecutor)
ssrmsd batch \
  --dir "af3=$FIXTURE/af3" \
  --dir "esmfold2=$FIXTURE/esmfold2" \
  --join intersection \
  --workers 2 \
  --align-chains A \
  --score-chains B,C \
  --ref-source af3 \
  --pairwise-out "$work/batch_pairs.csv" \
  -o "$work/batch_scores.csv"

# 4. Single ensemble score: AF3 .tar.gz (with decoy) + poly-G-linked .pdb
ssrmsd score \
  "af3=$FIXTURE/single/mab1_af3.tar.gz" \
  "esmfold=$FIXTURE/single/mab1_linked.pdb" \
  --id mab1_single \
  --align-chains A \
  --score-chains B,C \
  --ref-source af3 \
  --pairwise-out "$work/single_pairs.csv" \
  -o "$work/single_score.csv"

# 5. Manifest batch (JSONL, --workers 2) mixing a valid ensemble and a homomer error row
cat > "$work/manifest.csv" <<CSV
ensemble_id,path,source
mab1,$FIXTURE/af3/mab1,af3
mab1,$FIXTURE/esmfold2/mab1_sample_0.cif,esmfold2
mab1,$FIXTURE/esmfold2/mab1_sample_1.cif,esmfold2
hom1,$FIXTURE/homomer,af3
CSV

ssrmsd batch \
  --manifest "$work/manifest.csv" \
  --workers 2 \
  --align-chains A \
  --score-chains B,C \
  --ref-source af3 \
  --format jsonl \
  -o "$work/manifest_scores.jsonl"

# 6. Single-ensemble score on the homomer must exit 1 and record status=error
set +e
ssrmsd score "$FIXTURE/homomer" -o "$work/hom_score.csv"
hom_rc=$?
set -e
if [[ $hom_rc -ne 1 ]]; then
  echo "expected exit 1 on homomer single score, got $hom_rc" >&2
  exit 1
fi

# 7. Verify all outputs against analytical ground truth
python3 -I - "$FIXTURE/expected.json" "$work" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

expected = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
work = Path(sys.argv[2])

NUMERIC_COLS = (
    "mean_sq_rmsd",
    "rms_rmsd",
    "mean_rmsd",
    "median_rmsd",
    "rmsd_sd",
    "ssRMSD",
    "SD",
    "within_source_mean_rmsd",
    "cross_source_mean_rmsd",
)


def assert_metrics(
    row: dict, exp: dict, label: str, *, abs_tol: float = 1e-4, rel_tol: float = 0.0
) -> None:
    assert row["status"] == "ok", f"{label}: status={row['status']} error={row.get('error')}"
    assert int(row["n_structures"]) == exp["n_structures"], f"{label}: n_structures={row['n_structures']}"
    assert int(row["n_ca_align"]) == exp["n_ca_align"], f"{label}: n_ca_align={row['n_ca_align']}"
    assert int(row["n_ca_score"]) == exp["n_ca_score"], f"{label}: n_ca_score={row['n_ca_score']}"
    for col in NUMERIC_COLS:
        got = float(row[col])
        want = float(exp[col])
        assert math.isclose(got, want, rel_tol=rel_tol, abs_tol=abs_tol), (
            f"{label}.{col}: got {got}, expected {want}"
        )


# Batch --dir CSV + pairwise CSV
with (work / "batch_scores.csv").open(encoding="utf-8", newline="") as f:
    brows = list(csv.DictReader(f))
assert [r["ensemble_id"] for r in brows] == ["mab1", "mab2", "mab3"], brows
for row in brows:
    assert row["sources"] == "af3:3,esmfold2:2", row
    assert_metrics(row, expected[row["ensemble_id"]], f"batch/{row['ensemble_id']}")

with (work / "batch_pairs.csv").open(encoding="utf-8", newline="") as f:
    bprows = list(csv.DictReader(f))
assert len(bprows) == 30, f"expected 30 batch pairwise rows, got {len(bprows)}"

# Single score CSV + pairwise CSV (PDB coords truncate at 3 decimals -> ~1e-4 relative on squared sums)
with (work / "single_score.csv").open(encoding="utf-8", newline="") as f:
    srows = list(csv.DictReader(f))
assert len(srows) == 1
assert srows[0]["ensemble_id"] == "mab1_single"
assert srows[0]["sources"] == "af3:3,esmfold:1"
assert_metrics(srows[0], expected["mab1_single"], "single/mab1_single", abs_tol=1e-3, rel_tol=1e-4)

with (work / "single_pairs.csv").open(encoding="utf-8", newline="") as f:
    sprows = list(csv.DictReader(f))
assert len(sprows) == 6, f"expected 6 single pairwise rows, got {len(sprows)}"

# Manifest JSONL: preserves input order, ok + error
mrows = [json.loads(line) for line in (work / "manifest_scores.jsonl").read_text(encoding="utf-8").splitlines()]
assert [r["ensemble_id"] for r in mrows] == ["mab1", "hom1"], mrows
assert_metrics(mrows[0], expected["mab1"], "manifest/mab1")
assert mrows[1]["status"] == "error", mrows[1]
assert "homomer" in mrows[1]["error"].lower(), mrows[1]["error"]

# Homomer single score CSV
with (work / "hom_score.csv").open(encoding="utf-8", newline="") as f:
    hrows = list(csv.DictReader(f))
assert len(hrows) == 1 and hrows[0]["status"] == "error"
assert "homomer" in hrows[0]["error"].lower(), hrows[0]["error"]

print("ssrmsd test passed: 3 batch ensembles (30 pairs), 1 archive+polyG score (6 pairs), manifest ok+error, Pyxis isolation")
PY
