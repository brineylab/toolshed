set -euo pipefail
cd "$(dirname "$0")"
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(stamp)] $(hostname): $(nvidia-smi -L 2>/dev/null || echo 'no gpu visible')"
out=$(mktemp -d)
# The fixture is this campaign's build of STok's own CAMEO test structure
# 7YPD_B: shard 00000 holds two accepted 243-residue rows (one complete, one
# with a residue unobserved and one missing O, exercising the null-token and
# null-coordinate paths), shard 00001 a 20-residue row the fixed policy must
# reject -- well past the policy's 25-residue floor, under its 1280 ceiling.
start=$(date +%s)
echo "[$(stamp)] tokenizing the fixture on cuda"
stok tokenize-structures fixture "$out/tokens" --preset large \
  --text-column cif --id-column id \
  --keep-column mean_plddt --keep-column uniprot_accession \
  --include-coordinates --device cuda
echo "[$(stamp)] tokenization finished in $(( $(date +%s) - start ))s; checking the answer"
python3 -I check.py "$out/tokens"
echo "[$(stamp)] all checks passed, $(( $(date +%s) - start ))s from tokenize start"
