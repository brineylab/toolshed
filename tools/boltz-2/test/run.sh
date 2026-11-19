set -euo pipefail
cd "$(dirname "$0")"
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(stamp)] $(hostname): $(nvidia-smi -L 2>/dev/null || echo 'no gpu visible')"
out=$(mktemp -d)
inputs=$out/inputs
mkdir "$inputs"
cp complex.yaml complex_large.yaml "$inputs/"
# Both fixtures go through one boltz process: a directory input runs every
# yaml in it in a single invocation, so the 3 GB checkpoint loads once and the
# cuEquivariance Triton kernels compile once for the GPU at hand, not once per
# fixture. One diffusion sample per fixture: the tool test has a 30-minute
# budget that the image import and the checkpoint load also come out of, and
# the first complex past cuequivariance's 100-residue crossover pays the
# kernels' first-use Triton compile, the slowest step of a short run on a new
# architecture like Blackwell's sm_100. 1.0.1's test ran the two fixtures as
# two invocations at 5 samples each and passed on an L40S; that shape no
# longer fits the budget on a B200, whose 198-residue fixture was still inside
# its first compile when the limit hit.
echo "[$(stamp)] predicting both fixtures in one process, 1 diffusion sample each"
boltz predict "$inputs" --out_dir "$out" --diffusion_samples 1 --seed 0
echo "[$(stamp)] prediction finished; checking outputs"
python3 -I check.py "$out"
echo "[$(stamp)] all checks passed"
if [ -n "${FLYLOFT_OUT:-}" ] && [ -d "${FLYLOFT_OUT}" ]; then
  cp "$out"/boltz_results_inputs/predictions/*/confidence_*_model_0.json "${FLYLOFT_OUT}"/
fi
