set -euo pipefail
cd "$(dirname "$0")"
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(stamp)] $(hostname): $(nvidia-smi -L 2>/dev/null || echo 'no gpu visible')"
# Under pyxis the caller's environment comes in; the tool must still load what is in the image.
export HF_HOME=/tmp/nowhere HF_HUB_CACHE=/tmp/nowhere PYTHONPATH=/tmp/nowhere
out=$(mktemp -d)
echo "[$(stamp)] a job without a name is refused before the model loads"
if esmfold2 predict --in bad.json --out "$out/bad"; then echo "bad.json was accepted"; exit 1; fi
echo "[$(stamp)] folding the fixture's three jobs (the 198-residue HIV-1 protease dimer alone and with an MSA, GCN4-p1) in one process"
esmfold2 predict --in jobs.json --out "$out" --seed 0
echo "[$(stamp)] again into the same directory"
esmfold2 predict --in jobs.json --out "$out" --seed 0
echo "[$(stamp)] predictions finished; checking outputs"
python3 -I check.py "$out"
echo "[$(stamp)] all checks passed"
