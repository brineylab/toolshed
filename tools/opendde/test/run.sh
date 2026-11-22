set -euo pipefail
cd "$(dirname "$0")"
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(stamp)] $(hostname): $(nvidia-smi -L 2>/dev/null || echo 'no gpu visible')"
# Under pyxis the caller's environment comes in; the tool must still load what is in the image.
export HF_HOME=/tmp/nowhere HF_HUB_CACHE=/tmp/nowhere PYTHONPATH=/tmp/nowhere
out=$(mktemp -d)
echo "[$(stamp)] folding the 198-residue HIV-1 protease dimer, single-sequence, 1 sample"
opendde pred -i hivpr.json -o "$out" -n opendde_v1 --use_msa false --use_template false --use_rna_msa false --seeds 0 --sample 1
echo "[$(stamp)] prediction finished; checking outputs"
python3 -I check.py "$out"
echo "[$(stamp)] all checks passed"
