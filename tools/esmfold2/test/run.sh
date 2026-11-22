set -euo pipefail
cd "$(dirname "$0")"
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(stamp)] $(hostname): $(nvidia-smi -L 2>/dev/null || echo 'no gpu visible')"
# Under pyxis the caller's environment comes in; the tool must still load what is in the image.
export HF_HOME=/tmp/nowhere HF_HUB_CACHE=/tmp/nowhere
# A caller's PYTHONPATH holding a numpy that raises: the tool must not import it.
export PYTHONPATH=$(mktemp -d)
echo 'raise ImportError("the caller'"'"'s PYTHONPATH reached the tool")' > "$PYTHONPATH/numpy.py"
out=$(mktemp -d)
# Peak GPU memory for the skill's Compute: nvidia-smi sampled every half second through the run.
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$out/.gpu-mem" 2>/dev/null &
sampler=$!
echo "[$(stamp)] a job without a name is refused before the model loads"
if esmfold2 predict --in bad.json --out "$out/bad"; then echo "bad.json was accepted"; exit 1; fi
echo "[$(stamp)] an MSA row with an A3M's lowercase insertions left in is refused before the model loads"
if esmfold2 predict --in bad_msa.json --out "$out/bad"; then echo "bad_msa.json was accepted"; exit 1; fi
echo "[$(stamp)] folding the fixture's three jobs (the 198-residue HIV-1 protease dimer alone and with an MSA, GCN4-p1) in one process"
esmfold2 predict --in jobs.json --out "$out" --seed 0
echo "[$(stamp)] again into the same directory"
esmfold2 predict --in jobs.json --out "$out" --seed 0
echo "[$(stamp)] predictions finished; checking outputs"
python3 -I check.py "$out"
echo "[$(stamp)] all checks passed"
kill "$sampler" 2>/dev/null || true
echo "[$(stamp)] peak GPU memory: $(sort -n "$out/.gpu-mem" | tail -1) MiB"
