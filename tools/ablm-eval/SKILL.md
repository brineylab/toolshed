---
name: ablm-eval
description: "AbLM-Eval: score antibody datasets with the lab's antibody language models from one YAML config — per-position and whole-sequence log-likelihoods, mutation and naturalness prediction, MoE routing, and classification fine-tuning over CV folds. Reach for it to score, compare or fine-tune the lab's published models (BALM-paired, BALM-unpaired, ft-ESM, BALM-MoE, CurrAb, the preferential/uniform checkpoints, the 350M scaling model) on paired or single-chain antibody data."
metadata:
  version: 1.0.2
---

# ablm-eval

Image: `ghcr.io/brineylab/toolshed/ablm-eval:1.0.2`

AbLM-Eval (github.com/smburbach/AbLM-Eval at commit `32a3961`, MIT) runs its evaluation tasks on the lab's published antibody language models from one YAML config: inference, per-position inference, mutation prediction, naturalness, classification (fine-tuning a head over CV folds) and MoE routing. The nine lab models are registered by name, each pinned to a full Hugging Face commit and carrying the chain separator it was trained with. A general-purpose image (DESIGN §1): it carries no weights and sets no `HF_HUB_OFFLINE`, because it downloads the models a config names at run time, into `--cache-dir`.

## Commands

```bash
ablm-eval models
ablm-eval run config.yaml --cache-dir /work/ablm-eval-cache
ablm-eval run config.yaml --cache-dir /work/ablm-eval-cache --overwrite
```

`ablm-eval models` lists the registered models with their repository, revision and separator:

| name | repository | separator |
| --- | --- | --- |
| `balm-paired` | `brineylab/BALM-paired` | `</s>` |
| `balm-unpaired` | `brineylab/BALM-unpaired` | single chains only |
| `ft-esm` | `brineylab/ft-ESM` | `<cls><cls>` |
| `balm-moe` | `brineylab/BALM-MoE` | `<cls>` |
| `currab` | `brineylab/CurrAb` | `<cls>` |
| `preferential-250k` | `brineylab/preferential-250k` | `<cls><cls>` |
| `uniform-250k` | `brineylab/uniform-250k` | `<cls><cls>` |
| `uniform-350k` | `brineylab/uniform-350k` | `<cls><cls>` |
| `scaling-350m-full` | `brineylab/350M_full_checkpoint-500000` | `<cls><cls>` |

A config names any of them as a bare string, or spells a model out in full — `name`, `path` (a Hugging Face repository), `revision` (a full 40-character SHA), `separator` — which is how an unregistered checkpoint such as the 8M one below is evaluated. Configs run every task on every model, in order; with more than one model, results are compared after the last model finishes.

## Inputs and outputs

- **In:** a YAML config with `output_dir`, `models` and `tasks`. Tasks are keyed `inference`, `per_position`, `mutation_prediction`, `naturalness`, `classification`, `routing`, and share fields for `data_path` (CSV or parquet), `antibody_datatype` (`paired` or `unpaired`), `dataset_name` (names the task's output directory; distinct per task), `num_proc` (dataset preparation workers; default 128), and `dataset_columns` (overrides for the expected column names). Paired data defaults to `sequence_id`, `sequence_aa_heavy`, `sequence_aa_light`, with optional `cdr_mask_heavy`/`cdr_mask_light` (a `0`/`1` string per residue, CDRs `1`) and `v_mutation_count_aa_heavy`/`v_mutation_count_aa_light` (a `0`/`1` string per residue, mutated `1`); unpaired data defaults to `sequence_aa` with optional `locus`, `cdr_mask`, `v_mutation_count_aa`. A classification task takes `data_path` as `{fold: {train: path, test: path}}`, with `num_folds` (default 5) and `num_classes` (default 2); leave `training_args` unset to train on AbLM-Eval's defaults (see Gotchas). A small config, whole:

  ```yaml
  output_dir: results
  models:
    - name: tiny
      path: brineylab/8M_full_checkpoint-500000
      revision: 01e6532f909186a7f32ffd3e5fd6ced80daa0cc1
      separator: "<cls><cls>"
  tasks:
    - task: per_position
      dataset_name: mini
      antibody_datatype: paired
      data_path: /work/data/paired-mini.parquet
      num_proc: 1
  ```

- **Out:** `<output_dir>/models.json` records the run's models and separators; each task writes its config to `<output_dir>/<dataset_name>_<task>/config.json` and one parquet per model under `results/` (the per-position file is `<model>_<dataset_name>-per-position-inference.parquet`). With more than one model, comparison plots and summary CSVs land beside them. The per-position parquet keeps the dataset's own columns plus `sequence` (chains joined by the separator), `tokenized_sequence`, `tokenized_seq_wo_special`, `loss` (one cross-entropy per non-special position: each position masked in turn, the true residue's negative log-probability), `perplexity` (`exp(loss)` per position), `probabilities` (the full next-token distribution per position), `prediction_tokens` and `prediction` (the model's top prediction per position). The other tasks write the analogous columns for their metrics — `inference` one mean loss per sequence, `mutation_prediction` and `naturalness` their per-mutation scores, `classification` one CSV per model at `<output_dir>/<dataset_name>_classification/results/` with the fold's test metrics, `routing` the MoE expert assignments.

## Compute

One GPU per job: the models are small enough that one is faster than several through the DataParallel the code falls back to when it sees more. Per-position inference costs one forward pass per sequence position (each sequence becomes one batch of L masked copies), so its cost grows as sequence count times the square of length; the other inference tasks cost one pass per sequence. Measured on one b200 in a venv carrying the image's pinned package set on CPython 3.12, the image's Python — the image's twin, since the image cannot be measured before it publishes — downloading each model once into `--cache-dir` and running `per_position` on the 8-pair fixture (heavy chains 124-134 residues, light 107-113):

| model | separator | mean per-residue log-probability | wall time |
| --- | --- | --- | --- |
| `balm-paired` | `</s>` | -0.3063 | 18.5 s |
| `balm-unpaired` | single chains only | -0.3713 | 12.7 s |
| `ft-esm` | `<cls><cls>` | -0.2964 | 25.9 s |
| `balm-moe` | `<cls>` | -0.2848 | 15.9 s |
| `currab` | `<cls>` | -0.2723 | 20.1 s |
| `preferential-250k` | `<cls><cls>` | -0.2817 | 13.5 s |
| `uniform-250k` | `<cls><cls>` | -0.2924 | 12.8 s |
| `uniform-350k` | `<cls><cls>` | -0.2739 | 14.6 s |
| `scaling-350m-full` | `<cls><cls>` | -0.2915 | 14.4 s |

Each row spans the line that announces the model — printed just before its download — to its results parquet, so its weights download is inside the number. All nine load under AbLM-Eval's strict weight check and finish: the eight paired models ran in one config (154.9 s cold, to the final comparison), and `balm-unpaired` ran the same eight heavy chains as an unpaired task (29.6 s cold). The per-residue log-probabilities are deterministic — for the eight models the AbLM-Eval 32a3961 bump leaves alone they equal the 1.0.1 image's measurements — while wall times vary run to run.

Classification fine-tuning costs one forward and backward pass per training step plus one predict pass over the test fold. The HD-CoV check — fold 0 of the MoE paper's data (Zenodo record 19411192), one fold, through `ablm-eval run` on one GPU with AbLM-Eval's default accelerate config (DeepSpeed ZeRO 0, bf16, one process) and its default training arguments (no `training_args` in the task: warmup_steps 0.1, bf16, batch 32, 3 epochs of the fold's 20,257 rows, about 1,900 steps), measured in the same venv as the table above — trains and writes scores for both the 8M checkpoint (139.4 s wall from a cold cache, fetch to final metrics) and `balm-moe`, whose attention classifier is on by default (752.4 s); their test metrics on the fold's 5,065 test rows are from that short default-args run, not model quality: the 8M checkpoint's accuracy 0.7765, precision 0.7827, recall 0.7654, AUROC 0.8613, AUPR 0.8651, F1 0.7740, MCC 0.5531, and `balm-moe`'s accuracy 0.7822, precision 0.7816, recall 0.7832, AUROC 0.8569, AUPR 0.8374, F1 0.7824, MCC 0.5645.

The shed's test (`test/run.sh`) took 2.5 minutes of job wall on one b200 in the 1.0.1 candidate's run (image import, the registry check, the 8M checkpoint fetch, `per_position` and the reference comparison); the test is unchanged in 1.0.2, and its `--overwrite` exercise — the refused rerun, the recomputation, the comparison again — adds well under a minute, the checkpoint being already in the cache.

## Running it

On a laptop, from the directory holding `config.yaml` and the data (the tasks use the GPU when one is visible, and CPU otherwise, many times slower):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/ablm-eval:1.0.2 ablm-eval run config.yaml --cache-dir /work/cache
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/ablm-eval:1.0.2 \
  --container-mounts="$PWD":/work --container-workdir=/work \
  ablm-eval run config.yaml --cache-dir /work/cache
```

The model files land in `--cache-dir` (the default `.ablm-eval-cache` sits in the working directory); the nine registered models are between 30 MB and 2.9 GB each, so keep the cache on scratch and reuse it across runs — a rerun with the same `--cache-dir` downloads nothing.

## Gotchas

- **`--overwrite` reruns:** a task directory that already holds files is refused, and the run stops, unless `--overwrite` is passed — so after a failed or preempted run, rerun the same config with `--overwrite` to reuse the output directory (completed or partial results in it are recomputed, not skipped).
- **`balm-unpaired` is single-chain:** a `paired` task with it in the config is refused by design, naming it. Run it in its own config with `antibody_datatype: unpaired` and single-chain data.
- **The separator comes from the model:** each registered model carries the separator it was trained with and applies it to every task; setting `separator` in a task config alongside such a model is refused. Only full model entries with `separator` of their own, and `balm-unpaired` (which leaves the task's separator alone), differ.
- **`num_proc` defaults to 128** dataset-preparation workers; on a node with fewer cores, set it to what the node has.
- **Rich wraps at 80 columns** when a terminal is not a tty: export `COLUMNS=1000` before `ablm-eval models` in scripts, or the table's repository and revision columns wrap.
- **Classification's `attention_classifier` follows the model:** the task field is unset by default — a model whose config defines it (`balm-moe` only) gets it on, every other model gets nothing, and setting it on a model whose config does not define it is refused with a `ValueError` naming the model. A task with no `training_args` trains on AbLM-Eval's defaults (warmup_steps 0.1, bf16, batch 32, 3 epochs, report_to none); spell out `training_args` only to depart from them. One trap is the caller's, not AbLM-Eval's: under Slurm/pyxis the caller's environment is passed into the container, and an ambient `LOCAL_RANK`/`RANK` with no `WORLD_SIZE` kills a CLI validating a config whose task carries `training_args` — transformers' `TrainingArguments`, built while the config is validated, inits `torch.distributed` at the `env://` rendezvous and dies (`environment variable WORLD_SIZE expected, but not set`); unset them for the CLI, which is a single process that launches its own training through accelerate and sets the training subprocess's variables itself. A task with no `training_args` builds no `TrainingArguments` at validation, so it does not hit the trap.
- **The test** (`test/run.sh`) runs `per_position` through the CLI on AbLM-Eval's own fixture `test/paired-mini.parquet` — the 8 public pairs from Jaffe et al. 2022 (figshare 19617633), heavy chains 124-134 and light 107-113 residues, so each sequence builds a real masked batch of 107-134 rows — with the 8M checkpoint `brineylab/8M_full_checkpoint-500000` @ `01e6532f` spelled out as a full model entry with separator `<cls><cls>`, fetched at test time. It compares the pooled mean per-residue log-probability and each sequence's mean with `test/reference.json`, computed once with `evaluate_ablms` (AbLM-Eval's Python API) at commit `641cff8` on the same fixture, model and pinned versions, to 1e-4 — unchanged for 1.0.2: the AbLM-Eval bump touches no per-position inference path, and the reference still matches. It also checks every registered model's name is listed by `ablm-eval models`, and exercises the documented `--overwrite`: the same config rerun without it is refused (`The directory '...' exists and is not empty!`), and rerun with it the results are recomputed and the comparison passes again.
- **Licenses:** AbLM-Eval is MIT; every model the registry names (and the 8M test checkpoint) is MIT-licensed on the Hub. The image carries no weights — it downloads them at run time — so a run needs the network and a place to put `--cache-dir`.
