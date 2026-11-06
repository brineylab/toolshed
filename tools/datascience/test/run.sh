set -euo pipefail
rm -rf /tmp/abstar /tmp/mm*
abstar run /toolshed/test/pairs.fasta /tmp/abstar --germline_database human -o parquet
python3 -I - <<'EOF'
import glob
import polars as pl
df = pl.read_parquet(glob.glob("/tmp/abstar/**/*.parquet", recursive=True))
assert df.height == 600, f"{df.height} rows, not 600"
regions = ["fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa", "fwr4_aa"]
missing = {"sequence_vdj_aa", *regions} - set(df.columns)
assert not missing, f"missing columns {sorted(missing)}"
prod = df.filter(pl.col("productive"))
assert prod.height >= 570, f"{prod.height} productive, not at least 570"
bad = prod.filter(pl.col("sequence_vdj_aa") != pl.concat_str(regions))
assert bad.is_empty(), f"sequence_vdj_aa is not the joined regions for {bad['sequence_id'][0]}"
with open("/tmp/mm_heavy.fasta", "w") as f:
    for sid, aa in prod.filter(pl.col("locus") == "IGH").select("sequence_id", "sequence_vdj_aa").iter_rows():
        f.write(f">{sid}\n{aa}\n")
print(f"abstar: {df.height} rows, {prod.height} productive")
EOF
mmseqs easy-cluster /tmp/mm_heavy.fasta /tmp/mm /tmp/mmtmp --min-seq-id 0.85 -c 0.8 --threads 2 > /tmp/mm.log
clusters=$(cut -f1 /tmp/mm_cluster.tsv | sort -u | wc -l)
echo "mmseqs: $clusters clusters"
[[ $clusters -eq 44 ]]  # this mmseqs binary's answer on the fixture
diamond version | grep -q '^diamond version 2\.'
python3 -I -c "import abstar, abutils, numpy, polars, pandas, pyarrow, duckdb, scipy, sklearn, Bio, huggingface_hub, datasets, matplotlib, seaborn; assert abstar.__version__ == '0.9.0', abstar.__version__"
[[ "$(git config --system --get-all safe.directory)" == "*" ]]
[[ -z "${HF_HUB_OFFLINE:-}" ]]
echo ok
