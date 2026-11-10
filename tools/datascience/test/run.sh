set -euo pipefail
d=$(mktemp -d)
abstar run /toolshed/test/pairs.fasta "$d/abstar" --germline_database human -o parquet
python3 -I - "$d" <<'PY'
import glob, sys
import polars as pl
d = sys.argv[1]
df = pl.read_parquet(glob.glob(f"{d}/abstar/**/*.parquet", recursive=True))
assert df.height == 600, f"{df.height} rows, not 600"
regions = ["fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa", "fwr4_aa"]
missing = {"sequence_vdj_aa", "v_call", "d_call", "j_call", *regions} - set(df.columns)
assert not missing, f"missing columns {sorted(missing)}"
prod = df.filter(pl.col("productive"))
assert prod.height >= 570, f"{prod.height} productive, not at least 570"
bad = prod.filter(pl.col("sequence_vdj_aa") != pl.concat_str(regions))
assert bad.is_empty(), f"sequence_vdj_aa is not the joined regions for {bad['sequence_id'][0]}"
with open(f"{d}/heavy.fasta", "w") as f:
    for sid, aa in prod.filter(pl.col("locus") == "IGH").select("sequence_id", "sequence_vdj_aa").iter_rows():
        f.write(f">{sid}\n{aa}\n")
print(f"abstar: {df.height} rows, {prod.height} productive")
PY
anarcii=$(python3 -I - "$d" <<'PY'
import glob, sys
import polars as pl
from anarcii import Anarcii
d = sys.argv[1]
df = pl.read_parquet(glob.glob(f"{d}/abstar/**/*.parquet", recursive=True)).filter(pl.col("productive"))
rows = {sid: (aa, locus, cdr3) for sid, aa, locus, cdr3 in df.select("sequence_id", "sequence_vdj_aa", "locus", "cdr3_aa").iter_rows()}
numbered = Anarcii(seq_type="antibody", mode="accuracy", cpu=True, ncpu=2).number({sid: r[0] for sid, r in rows.items()})
loci = {"H": "IGH", "K": "IGK", "L": "IGL"}
chains = cdr3s = 0
for sid, (aa, locus, cdr3) in rows.items():
    n = numbered[sid]
    chains += loci.get(n["chain_type"]) == locus
    cdr3s += "".join(r for (pos, _), r in n["numbering"] if 105 <= pos <= 117 and r != "-") == cdr3
print(f"{len(rows)} {chains} {cdr3s}")
PY
)
read -r n chains cdr3s <<< "$anarcii"
echo "anarcii: $chains of $n chain types and $cdr3s of $n CDR3s agree with abstar"
[[ $n -eq 599 && $chains -eq 599 && $cdr3s -eq 599 ]]  # ANARCII 2.0.8 agrees with abstar 0.9.0 on every productive chain; IMGT 111A, 112A ... insertions included
mmseqs easy-cluster "$d/heavy.fasta" "$d/mm" "$d/mmtmp" --min-seq-id 0.85 -c 0.8 --threads 2 > "$d/mm.log"
clusters=$(cut -f1 "$d/mm_cluster.tsv" | sort -u | wc -l)
echo "mmseqs: $clusters clusters"
[[ $clusters -eq 39 ]]  # MMseqs2 2a3dafc's answer on the fixture; release 18 and abutils' bundled build give 44
diamond version | grep -q '^diamond version 2\.'
diamond makedb --in "$d/heavy.fasta" -d "$d/heavy" --quiet
diamond blastp -d "$d/heavy.dmnd" -q "$d/heavy.fasta" -o "$d/hits.tsv" --threads 2 --quiet
queries=$(grep -c '>' "$d/heavy.fasta")
selfhits=$(awk '$1 == $2' "$d/hits.tsv" | cut -f1 | sort -u | wc -l)
echo "diamond: $selfhits of $queries heavy chains find themselves"
[[ $selfhits -eq $queries ]]
python3 -I -c "import abstar, abutils, anarcii, torch, numpy, polars, pandas, pyarrow, duckdb, scipy, sklearn, Bio, huggingface_hub, datasets, matplotlib, seaborn; assert abstar.__version__ == '0.9.0', abstar.__version__; assert anarcii.__version__ == '2.0.8', anarcii.__version__"
[[ "$(git config --system --get-all safe.directory)" == "*" ]]
[[ -z "${HF_HUB_OFFLINE:-}" ]]
echo ok
