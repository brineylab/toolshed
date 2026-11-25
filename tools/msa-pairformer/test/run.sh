set -euo pipefail
cd "$(dirname "$0")"
echo "node: $(hostname)"
nvidia-smi -L
export HF_HOME=/tmp/nowhere HF_HUB_CACHE=/tmp/nowhere
export PYTHONPATH=/tmp/does-not-exist
out=$(mktemp -d)
mkdir "$out/input"
gzip -c fixture.a3m > "$out/input/fixture.a3m.gz"
# Exercise real multi-row hhfilter and greedy diversity selection past the
# cuEquivariance 100-residue threshold, plus query-only masks and resumption.
command=(msa-pairformer --in "$out/input" --depth 16 --k 7 --query-identity-cap .99 --min-coverage .82)
"${command[@]}" --out "$out/result"
"${command[@]}" --out "$out/result"
cp -a "$out/result" "$out/resumed"
"${command[@]}" --out "$out/resumed"
cmp "$out/result/metrics.jsonl" "$out/resumed/metrics.jsonl"
python3 -I check.py "$out/result" 7

msa-pairformer --in "$out/input" --out "$out/two-fold" --depth 16 --k 2 --query-identity-cap .99 --min-coverage .82
python3 -I check.py "$out/two-fold" 2 "$out/result"

# Exercise the documented eight-way sharding on separate per-job parents.
# Sorted inputs put fixture in job 0 and other in job 1; rerunning each job must
# validate and skip its completed chain without changing any output artifact.
mkdir "$out/sharded_input"
cp "$out/input/fixture.a3m.gz" "$out/sharded_input/fixture.a3m.gz"
cp "$out/input/fixture.a3m.gz" "$out/sharded_input/other.a3m.gz"
sharded=(msa-pairformer --in "$out/sharded_input" --depth 16 --k 7 --query-identity-cap .99 --min-coverage .82 --num-jobs 8)
for index in 0 1; do
    job_out="$out/shards/job_$index"
    "${sharded[@]}" --out "$job_out" --job-index "$index"
    before="$out/before_resume_$index"
    cp -a "$job_out" "$before"
    "${sharded[@]}" --out "$job_out" --job-index "$index"
    diff -r "$before" "$job_out"
done
python3 -I check.py "$out/shards/job_0" 7
python3 -I check.py "$out/shards/job_1" 7 "$out/result" other

echo "all checks passed; no network required"
