set -euo pipefail
w=$(mktemp -d)
cd "$w"
run() { local n=$1; shift; "$@" > "$w/$n.log" 2>&1 || { sed 's/\x1b\[[0-9;]*m//g' "$w/$n.log" | tail -40; exit 1; }; sed -i 's/\x1b\[[0-9;]*m//g' "$w/$n.log"; }
py() { /opt/venv/bin/python3 -I - "$@"; }

# Purity and completeness of an inferred partition against a simulation's true families: the share of sequences in
# the dominant true family of their inferred cluster, and in the dominant inferred cluster of their true family.
cat > "$w/score.py" <<'PY'
import sys
from collections import Counter
from partis import utils
def score(inferred, truth):
    fam = {u: i for i, a in enumerate(utils.read_output(truth)[1]) for u in a["unique_ids"]}
    clusters = [[u for u in c if u in fam] for c in utils.read_output(inferred)[2].best()]
    n = sum(len(c) for c in clusters)
    purity = sum(Counter(fam[u] for u in c).most_common(1)[0][1] for c in clusters if c) / n
    where = {u: i for i, c in enumerate(clusters) for u in c}
    by_fam = {}
    for u in where:
        by_fam.setdefault(fam[u], []).append(where[u])
    completeness = sum(Counter(v).most_common(1)[0][1] for v in by_fam.values()) / n
    return n, purity, completeness
PY
score() { py "$w" "$@" <<'PY'
import sys; sys.path.insert(0, sys.argv[1]); from score import score
n, p, c = score(*sys.argv[2:])
print(f"{n} {p:.3f} {c:.3f}")
PY
}

# Simulate IGH from scratch with the default per-base mutation model: trees from TreeSim in R, mutation along them by
# bpp-newlik's bppseqgen. partis falls back on nothing here: without R or bpp-newlik it fails.
run simu partis simulate --rearrange-from-scratch --outfname "$w/simu.yaml" --n-sim-events 20 --seed 1 --n-procs 2 --debug 1
[[ $(awk '/generating [0-9]+ trees/ { n += $2 } END { print n }' "$w/simu.log") -eq 20 ]]  # one batch per process
[[ $(grep -c 'using bpp per-base mutation with path /opt/partis/packages/bpp-newlik/_build' "$w/simu.log") -eq 20 ]]

# Parse it to FASTA, cache parameters and annotate, writing AIRR; check against the truth the simulation carries.
run simufa parse-output.py "$w/simu.yaml" "$w/seqs.fa"
run annotate partis annotate --infname "$w/seqs.fa" --parameter-dir "$w/params" --outfname "$w/ann.yaml" --airr-output --n-procs 2
[[ -d "$w/params/hmm" && -s "$w/ann.tsv" ]]
py "$w" <<'PY'
import csv, sys
from partis import utils
w = sys.argv[1]
_, simu, _ = utils.read_output(f"{w}/simu.yaml")
true_v = {u: a["v_gene"].split("*")[0] for a in simu for u in a["unique_ids"]}
rows = list(csv.DictReader(open(f"{w}/ann.tsv"), delimiter="\t"))
same = sum(r["v_call"].split("*")[0] == true_v[r["sequence_id"]] for r in rows)
mutated = sum(float(r["v_identity"]) < 1 for r in rows)
print(f"annotate: {len(rows)} sequences, {same} with the true V gene, {mutated} mutated in V")
assert len(rows) == len(true_v), (len(rows), len(true_v))
assert same >= 0.9 * len(rows) and mutated >= 0.8 * len(rows)
PY

# Partition them into clonal families, single chain.
run partition partis partition --infname "$w/seqs.fa" --parameter-dir "$w/params" --outfname "$w/partition.yaml" --n-procs 2
read -r n p c <<< "$(score "$w/partition.yaml" "$w/simu.yaml")"
echo "partition: $n IGH sequences, purity $p, completeness $c"
awk -v p="$p" -v c="$c" 'BEGIN { exit !(p >= 0.95 && c >= 0.75) }'

# Simulate again from the parameters inferred from that sample, mimicking it.
run simu2 partis simulate --parameter-dir "$w/params" --outfname "$w/simu2.yaml" --n-sim-events 10 --seed 2 --n-procs 2 --debug 1
[[ $(grep -c 'using bpp per-base mutation' "$w/simu2.log") -eq 10 ]]

# Paired heavy/light: simulate, partition from the mixed FASTA and its pairing, and score each chain against the truth.
# partition is not deterministic, and families of a few sequences make completeness coarse, so the bar is purity, and
# that pairing never leaves a chain less pure than single-chain partitioning does. 80 families give IGL about 100
# sequences: at 40 (56 IGL), parameter inference sometimes pruned a J gene one family needed, which then failed
# annotation and took its heavy partners with it (45 of 56 IGL kept). Here at least 85% of each chain must be kept.
run psimu partis simulate --paired-loci --rearrange-from-scratch --paired-outdir "$w/psimu" --n-sim-events 80 --seed 3 --n-procs 2
run ppartition partis partition --paired-loci --infname "$w/psimu/all-seqs.fa" --input-metafnames "$w/psimu/meta.yaml" --paired-outdir "$w/ppart" --n-procs 2
for l in igh igk igl; do
  read -r n p c <<< "$(score "$w/ppart/partition-$l.yaml" "$w/psimu/$l.yaml")"
  read -r _ ps _ <<< "$(score "$w/ppart/single-chain/partition-$l.yaml" "$w/psimu/$l.yaml")"
  t=$(grep -c "^>.*-$l\$" "$w/psimu/all-seqs.fa")
  echo "paired partition $l: $n of $t sequences, purity $p (alone $ps), completeness $c"
  awk -v n="$n" -v t="$t" -v p="$p" -v c="$c" -v ps="$ps" 'BEGIN { exit !(n >= 0.85 * t && p >= 0.95 && c >= 0.75 && p >= ps) }'
done

# Selection metrics on the paired partition, and parsing partis output into FASTA and AIRR.
run metrics partis get-selection-metrics --paired-loci --paired-outdir "$w/ppart" --n-procs 2
run fasta parse-output.py "$w/ppart/partition-igh.yaml" "$w/igh.fa"
run airr parse-output.py "$w/ppart/partition-igh.yaml" "$w/igh.tsv" --airr-output
py "$w" <<'PY'
import sys, yaml
from partis import utils
w = sys.argv[1]
n = sum(len(c) for c in utils.read_output(f"{w}/ppart/partition-igh.yaml")[2].best())
metrics = yaml.safe_load(open(f"{w}/ppart/selection-metrics.yaml"))
assert metrics and all("lb" in m for m in metrics), "no selection metrics"
fasta = sum(l.startswith(">") for l in open(f"{w}/igh.fa"))
airr = sum(1 for _ in open(f"{w}/igh.tsv")) - 1
print(f"selection metrics for {len(metrics)} families; parse-output wrote {fasta} FASTA and {airr} AIRR rows of {n} IGH sequences")
assert fasta == airr == n
PY
echo ok
