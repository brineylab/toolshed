set -euo pipefail
cd "$(dirname "$0")"
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(stamp)] $(hostname): $(nvidia-smi -L 2>/dev/null || echo 'no gpu visible')"
# Blackwell: past cuequivariance's 100-residue crossover the model runs its
# Triton kernels, and their first-use compile for sm_100 did not finish
# inside 29 minutes in two runs of this test on a B200 (runs fd77eed9 and
# 50bcfb7f, both cancelled at the 30-minute limit mid-prediction), so the
# tool-test budget cannot cover them. Raise both crossover thresholds past
# the fixtures' size: boltz then runs its plain-PyTorch ops -- the same code
# `--no_kernels` runs and pre-Ampere GPUs always run, no compile involved.
# The skill's Gotchas document the thresholds and the compile cost for
# complexes that keep the kernels.
export CUEQ_TRIMUL_FALLBACK_THRESHOLD=1000000
export CUEQ_TRIATTN_FALLBACK_THRESHOLD=1000000
out=$(mktemp -d)
inputs=$out/inputs
mkdir "$inputs"
cp complex.yaml complex_large.yaml "$inputs/"
# Both fixtures go through one boltz process: a directory input runs every
# yaml in it in a single invocation, so the 3 GB checkpoint loads once.
# One diffusion sample per fixture: the tool test has a 30-minute budget
# that the image import and the checkpoint load also come out of.
# 1.0.1's test ran the two fixtures as two invocations at 5 samples each and
# passed on an L40S; that shape no longer fits the budget on a B200.
echo "[$(stamp)] predicting both fixtures in one process, 1 diffusion sample each"
boltz predict "$inputs" --out_dir "$out" --diffusion_samples 1 --seed 0
echo "[$(stamp)] prediction finished; checking outputs"
python3 -I check.py "$out"
echo "[$(stamp)] all checks passed"
if [ -n "${FLYLOFT_OUT:-}" ] && [ -d "${FLYLOFT_OUT}" ]; then
  cp "$out"/boltz_results_inputs/predictions/*/confidence_*_model_0.json "${FLYLOFT_OUT}"/
fi
