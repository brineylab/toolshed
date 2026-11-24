set -euo pipefail
cd "$(dirname "$0")"
echo "node: $(hostname)"
nvidia-smi -L
export HF_HOME=/tmp/nowhere HF_HUB_CACHE=/tmp/nowhere
export PYTHONPATH=/tmp/does-not-exist
out=$(mktemp -d)
mkdir "$out/input"
gzip -c fixture.a3m > "$out/input/fixture.a3m.gz"
# Exercise the real multi-row hhfilter, all seven query-only masks, and both
# ordinary and moved-directory resumes on CUDA 13 cuEquivariance.
command=(msa-pairformer --in "$out/input" --depth 16 --k 7 --query-identity-cap .99 --min-coverage .82)
"${command[@]}" --out "$out/result"
"${command[@]}" --out "$out/result"
cp -a "$out/result" "$out/resumed"
"${command[@]}" --out "$out/resumed"
cmp "$out/result/metrics.jsonl" "$out/resumed/metrics.jsonl"
python3 -I check.py "$out/result" 7

msa-pairformer --in "$out/input" --out "$out/two-fold" --depth 16 --k 2 --query-identity-cap .99 --min-coverage .82
python3 -I check.py "$out/two-fold" 2 "$out/result"
echo "all checks passed; no network required"
