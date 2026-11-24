set -euo pipefail
cd "$(dirname "$0")"
echo "node: $(hostname)"
nvidia-smi -L
export HF_HOME=/tmp/nowhere HF_HUB_CACHE=/tmp/nowhere
export PYTHONPATH=/tmp/does-not-exist
out=$(mktemp -d)
mkdir "$out/input"
gzip -c fixture.a3m > "$out/input/fixture.a3m.gz"
# The 120-residue, 40-row synthetic MSA takes the same multi-row hhfilter and
# all seven masking passes as a real protein, and invokes CUDA 13 cuEquivariance.
msa-pairformer --in "$out/input" --out "$out/result" --depth 16 --k 7 --query-identity-cap 0.99 --min-coverage .82
msa-pairformer --in "$out/input" --out "$out/result" --depth 16 --k 7 --query-identity-cap 0.99 --min-coverage .82
python3 -I check.py "$out/result"
echo "all checks passed; no network required"
