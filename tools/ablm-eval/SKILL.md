---
name: ablm-eval
description: "AbLM-Eval: score antibody datasets with the lab's antibody language models from one YAML config — per-position and whole-sequence log-likelihoods, mutation and naturalness prediction, MoE routing, and classification fine-tuning over CV folds (classification is broken on the pinned versions; see Gotchas). Reach for it to score or compare the lab's published models (BALM-paired, BALM-unpaired, ft-ESM, BALM-MoE, CurrAb, the preferential/uniform checkpoints, the 350M scaling model) on paired or single-chain antibody data."
metadata:
  version: 1.0.1
---

# ablm-eval

Image: `ghcr.io/brineylab/toolshed/ablm-eval:1.0.1`

AbLM-Eval (github.com/smburbach/AbLM-Eval at commit `641cff8`, MIT) runs its evaluation tasks on the lab's published antibody language models from one YAML config. The nine lab models are registered by name, each pinned to a full Hugging Face commit and carrying the chain separator it was trained with. A general-purpose image (DESIGN §1): it carries no weights and sets no `HF_HUB_OFFLINE`, because it downloads the models a config names at run time, into `--cache-dir`.

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

- **In:** a YAML config with `output_dir`, `models` and `tasks`. Tasks are keyed `inference`, `per_position`, `mutation_prediction`, `naturalness`, `classification`, `routing`, and share fields for `data_path` (CSV or parquet), `antibody_datatype` (`paired` or `unpaired`), `dataset_name` (names the task's output directory; distinct per task), `num_proc` (dataset preparation workers; default 128), and `dataset_columns` (overrides for the expected column names). Paired data defaults to `sequence_id`, `sequence_aa_heavy`, `sequence_aa_light`, with optional `cdr_mask_heavy`/`cdr_mask_light` (a `0`/`1` string per residue, CDRs `1`) and `v_mutation_count_aa_heavy`/`v_mutation_count_aa_light` (a `0`/`1` string per residue, mutated `1`); unpaired data defaults to `sequence_aa` with optional `locus`, `cdr_mask`, `v_mutation_count_aa`. A small config, whole:

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

- **Out:** `<output_dir>/models.json` records the run's models and separators; each task writes its config to `<output_dir>/<dataset_name>_<task>/config.json` and one parquet per model under `results/` (the per-position file is `<model>_<dataset_name>-per-position-inference.parquet`). With more than one model, comparison plots and summary CSVs land beside them. The per-position parquet keeps the dataset's own columns plus `sequence` (chains joined by the separator), `tokenized_sequence`, `tokenized_seq_wo_special`, `loss` (one cross-entropy per non-special position: each position masked in turn, the true residue's negative log-probability), `perplexity` (`exp(loss)` per position), `probabilities` (the full next-token distribution per position), `prediction_tokens` and `prediction` (the model's top prediction per position). The other tasks write the analogous columns for their metrics — `inference` one mean loss per sequence, `mutation_prediction` and `naturalness` their per-mutation scores, `classification` per-fold metrics, `routing` the MoE expert assignments.

## Compute

One GPU per job: the models are small enough that one is faster than several through the DataParallel the code falls back to when it sees more. Per-position inference costs one forward pass per sequence position (each sequence becomes one batch of L masked copies), so its cost grows as sequence count times the square of length; the other inference tasks cost one pass per sequence. Measured inside the image on one b200, downloading each model once into `--cache-dir` and running `per_position` on the 8-pair fixture (heavy chains 124-134 residues, light 107-113):

| model | separator | mean per-residue log-probability | wall time |
| --- | --- | --- | --- |
| `balm-paired` | `</s>` | -0.3063 | 16.9 s |
| `balm-unpaired` | single chains only | -0.3713 | 12.6 s |
| `ft-esm` | `<cls><cls>` | -0.2964 | 26.2 s |
| `balm-moe` | `<cls>` | does not load (see below) | 1.3 s |
| `currab` | `<cls>` | -0.2723 | 21.0 s |
| `preferential-250k` | `<cls><cls>` | -0.2817 | 14.8 s |
| `uniform-250k` | `<cls><cls>` | -0.2924 | 15.9 s |
| `uniform-350k` | `<cls><cls>` | -0.2739 | 17.5 s |
| `scaling-350m-full` | `<cls><cls>` | -0.2915 | 19.3 s |

Each row spans the line that announces the model — printed just before its download — to its results parquet, so its weights download is inside the number. `balm-paired` and `ft-esm` ran inside the one-config run of the eight paired models; that run stops at the first model that fails, and it stopped at `balm-moe`, the third of its eight, 56.4 s in, so `currab` through `scaling-350m-full` each ran in a config of their own. `balm-unpaired` ran the same eight heavy chains as an unpaired task, that run 19.5 s. `balm-moe`'s row is its own-config retry, which found its weights already fetched: it does not load — transformers 5.23 hands its special tokens down to BALM's `BalmTokenizer` as `AddedToken` objects, and the `WordLevel` that tokenizer builds on the pinned tokenizers 0.23.3 wants a plain `str` (`TypeError: 'AddedToken' object is not an instance of 'str'`, while processing `unk_token`) — so it is reported as a failed row, not dropped or worked around.

Classification fine-tuning would cost one forward and backward pass per training step plus one predict pass over the test fold, but on the pinned versions it never reaches one: the HD-CoV check — fold 0 of the MoE paper's data (Zenodo record 19411192), one fold, 10 training steps, the 8M checkpoint, through `ablm-eval run` on one GPU with AbLM-Eval's default accelerate config — died at model construction on every attempt (22.0 s, 19.9 s and 27.5 s wall inside the image for the 8M checkpoint, the same with the task's `attention_classifier: false`, and `balm-paired`), before a single training step; the gotcha below says why.

The shed's test (`test/run.sh`) took 2.5 minutes of job wall on one b200 in the candidate's run (image import, the registry check, the 8M checkpoint fetch, `per_position` and the reference comparison); its `--overwrite` exercise — the refused rerun, the recomputation, the comparison again — adds well under a minute, the checkpoint being already in the cache.

## Running it

On a laptop, from the directory holding `config.yaml` and the data (the tasks use the GPU when one is visible, and CPU otherwise, many times slower):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/ablm-eval:1.0.1 ablm-eval run config.yaml --cache-dir /work/cache
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/ablm-eval:1.0.1 \
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
- **Classification (does not work on the pinned versions):** it fine-tunes a head with transformers' `Trainer` through accelerate — run non-interactively it uses accelerate's default config (DeepSpeed ZeRO 0, bf16, one process) without prompting — but it dies before training: AbLM-Eval's train script passes `attention_classifier` into `AutoModelForSequenceClassification.from_pretrained` unconditionally, and on transformers 5.23 a kwarg that is neither a recognized config field nor an attribute of the checkpoint's config is forwarded to the model constructor, so every model that loads here (each a plain esm or roberta config) fails with `TypeError: ... got an unexpected keyword argument 'attention_classifier'`. The HD-CoV check tried it three ways on one GPU — the 8M checkpoint, the same with the task's `attention_classifier: false` (the kwarg is passed whatever its value), and `balm-paired`, whose published config.json is a plain roberta config — and all three failed identically (22.0 s, 19.9 s and 27.5 s wall, inside the image). The only registered model whose config defines the field is `balm-moe`, which fails earlier at its tokenizer (the failed row above). A classification task with no `training_args` dies even earlier, at `TrainingArguments`: transformers 5.23 removed `warmup_ratio`, which AbLM-Eval's fallback still passes — spell the warmup as `warmup_steps` (an int, or a float in [0, 1) for the ratio). Fixing classification needs a change in AbLM-Eval itself — pass `attention_classifier` only when the checkpoint's config defines it, or stop passing it — so until upstream changes, use the image's other tasks. One trap is the caller's, not AbLM-Eval's: under Slurm/pyxis the caller's environment is passed into the container, and an ambient `LOCAL_RANK`/`RANK` with no `WORLD_SIZE` makes transformers' `TrainingArguments` — built while the CLI validates the config, before any task — init `torch.distributed` at the `env://` rendezvous and die (`environment variable WORLD_SIZE expected, but not set`); unset them for the CLI, which is a single process that launches its own training through accelerate, and sets the training subprocess's variables itself.
- **The test** (`test/run.sh`) runs `per_position` through the CLI on AbLM-Eval's own fixture `test/paired-mini.parquet` — the 8 public pairs from Jaffe et al. 2022 (figshare 19617633), heavy chains 124-134 and light 107-113 residues, so each sequence builds a real masked batch of 107-134 rows — with the 8M checkpoint `brineylab/8M_full_checkpoint-500000` @ `01e6532f` spelled out as a full model entry with separator `<cls><cls>`, fetched at test time. It compares the pooled mean per-residue log-probability and each sequence's mean with `test/reference.json`, computed once with `evaluate_ablms` (AbLM-Eval's Python API) at commit `641cff8` on the same fixture, model and pinned versions, to 1e-4. It also checks every registered model's name is listed by `ablm-eval models`, and exercises the documented `--overwrite`: the same config rerun without it is refused (`The directory '...' exists and is not empty!`), and rerun with it the results are recomputed and the comparison passes again.
- **Licenses:** AbLM-Eval is MIT; every model the registry names (and the 8M test checkpoint) is MIT-licensed on the Hub. The image carries no weights — it downloads them at run time — so a run needs the network and a place to put `--cache-dir`.
