set -euo pipefail
d=$(mktemp -d)
# the caches stay under /tmp: the model is fetched at test time and nothing is
# baked into the image; a general-purpose image reaches the network
export HF_HOME="$d/hf-home" MPLCONFIGDIR="$d/mpl"
[[ -z "${HF_HUB_OFFLINE:-}" ]]
[[ -z "${TRANSFORMERS_OFFLINE:-}" ]]

# the registry of the lab's published models, with revisions and separators
COLUMNS=1000 ablm-eval models | tee "$d/models.txt"
for m in balm-paired balm-unpaired ft-esm balm-moe currab preferential-250k uniform-250k uniform-350k scaling-350m-full; do
  grep -q "$m" "$d/models.txt"
done

# per-position inference through the CLI: the 8M checkpoint as a full model
# entry, separator <cls><cls>, AbLM-Eval's own fixture (8 public pairs from
# Jaffe et al. 2022); everything the run writes stays under $d
cat > "$d/run.yaml" <<YAML
output_dir: $d/results
models:
  - name: tiny
    path: brineylab/8M_full_checkpoint-500000
    revision: 01e6532f909186a7f32ffd3e5fd6ced80daa0cc1
    separator: "<cls><cls>"
tasks:
  - task: per_position
    dataset_name: mini
    antibody_datatype: paired
    data_path: /toolshed/test/paired-mini.parquet
    num_proc: 1
YAML
ablm-eval run "$d/run.yaml" --cache-dir "$d/cache"
python3 -I /toolshed/test/check.py "$d/results"
