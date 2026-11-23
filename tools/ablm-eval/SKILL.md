---
name: ablm-eval
description: "AbLM-Eval: score antibody datasets with the lab's antibody language models from one YAML config — per-position and whole-sequence log-likelihoods, mutation and naturalness prediction, MoE routing, and classification fine-tuning over CV folds. Reach for it to score, compare or fine-tune the lab's published models (BALM-paired, BALM-unpaired, ft-ESM, BALM-MoE, CurrAb, the preferential/uniform checkpoints, the 350M scaling model) and ablm-forge checkpoints on paired or single-chain antibody data."
metadata:
  version: 1.0.4
---

# ablm-eval

Image: `ghcr.io/brineylab/toolshed/ablm-eval:1.0.4`

AbLM-Eval (github.com/smburbach/AbLM-Eval at commit `13062db`, MIT, installed with its `forge` extra) runs its evaluation tasks on the lab's published antibody language models from one YAML config: inference, per-position inference, mutation prediction, naturalness, classification (fine-tuning a head over CV folds) and MoE routing. The nine lab models are registered by name, each pinned to a full Hugging Face commit and carrying the chain separator it was trained with. Checkpoints written by ablm-forge (github.com/brineylab/ablm-forge; this image carries ablm-forge 0.0.1 at commit `6da77a69db57afd84561e918f0b375eec9349817`, the one current when 1.0.4 was built) load too, through the Python API (see Gotchas). The packages are AbLM-Eval 13062db's own resolve on Python 3.12 — numpy 2.5.5, abutils 0.6.1, torch 2.13.0, transformers 5.17.0, pyarrow 20.0.0, BALM at AbLM-Eval's pin `a91d7b4` — every one pinned in `requirements.txt`. A general-purpose image (DESIGN §1): it carries no weights and sets no `HF_HUB_OFFLINE`, because it downloads the models a config names at run time, into `--cache-dir`.

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

A config names any of them as a bare string, or spells a model out in full — `name`, `path` (a Hugging Face repository), `revision` (a full 40-character SHA), `separator` — which is how an unregistered checkpoint such as the 8M one below is evaluated. A checkpoint on disk, an ablm-forge snapshot among them, is scored through the Python API, which takes a path:

```bash
python3 -c "from ablm_eval import PerPositionConfig, evaluate_ablms; evaluate_ablms({'step-500000': '/work/run/checkpoint-500000'}, [PerPositionConfig(data_path='/work/data/paired-mini.parquet', antibody_datatype='paired', dataset_name='mini', separator='<cls><cls>', num_proc=1)], shared_output_dir='/work/results')"
```

with the `separator` the checkpoint was trained with. Configs run every task on every model, in order; with more than one model, results are compared after the last model finishes.

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

One GPU per job: the models are small enough that one is faster than several through the DataParallel the code falls back to when it sees more. Per-position inference costs one forward pass per sequence position (each sequence becomes one batch of L masked copies), so its cost grows as sequence count times the square of length; the other inference tasks cost one pass per sequence. Measured in the image's twin — a venv of the same pinned packages on the same CPython 3.12, built from this tool's `requirements.txt`, because the image cannot be measured before it publishes — on one B200, downloading each model once into `--cache-dir` and running `per_position` on the 8-pair fixture (heavy chains 124-134 residues, light 107-113):

| model | separator | mean per-residue log-probability | wall time |
| --- | --- | --- | --- |
| `balm-paired` | `</s>` | -0.3063 | 26.8 s |
| `balm-unpaired` | single chains only | -0.3713 | 22.2 s |
| `ft-esm` | `<cls><cls>` | -0.2964 | 31.9 s |
| `balm-moe` | `<cls>` | -0.2827 | 15.2 s |
| `currab` | `<cls>` | -0.2723 | 19.9 s |
| `preferential-250k` | `<cls><cls>` | -0.2817 | 15.8 s |
| `uniform-250k` | `<cls><cls>` | -0.2924 | 15.4 s |
| `uniform-350k` | `<cls><cls>` | -0.2739 | 14.9 s |
| `scaling-350m-full` | `<cls><cls>` | -0.2915 | 15.1 s |

Each row spans the line that announces the model — printed just before its download — to its results parquet, so its weights download is inside the number. All nine load under AbLM-Eval's strict weight check and finish: the eight paired models ran in one config (174.5 s cold, to the final comparison), and `balm-unpaired` ran the same eight heavy chains as an unpaired task (44.6 s cold). The per-residue log-probabilities are reproducible across the numpy-2 re-resolve — for eight models they equal 1.0.3's in-image measurements to the fourth decimal — while wall times vary run to run. `balm-moe` is the exception, as before: -0.2827 here, -0.2821 in the 1.0.3 image and -0.2848 in 1.0.2's twin, so treat its per-residue log-probability as good to about 1e-3.

Classification fine-tuning costs one forward and backward pass per training step plus one predict pass over the test fold. The HD-CoV check — fold 0 of the MoE paper's data (Zenodo record 19411192), one fold, through `ablm-eval run` on one GPU with AbLM-Eval's default accelerate config (DeepSpeed ZeRO 0, bf16, one process) and its default training arguments (no `training_args` in the task: warmup_steps 0.1, bf16, batch 32, 3 epochs of the fold's 20,257 rows, about 1,900 steps), measured in the same twin as the table above — trains and writes scores for both the 8M checkpoint (167.2 s wall from a cold cache, fetch to final metrics) and `balm-moe`, whose attention classifier is on by default (777.8 s); their test metrics on the fold's 5,065 test rows are from that short default-args run, not model quality, and vary run to run with training's shuffling: the 8M checkpoint's accuracy 0.7753, precision 0.7735, recall 0.7784, AUROC 0.8635, AUPR 0.8687, F1 0.7760, MCC 0.5507, and `balm-moe`'s accuracy 0.7887, precision 0.7829, recall 0.7990, AUROC 0.8539, AUPR 0.8342, F1 0.7909, MCC 0.5776.

The shed's test (`test/run.sh`) took 137.7 s of wall on one B200 in the twin — the registry check, the 8M checkpoint fetch, `per_position`, the reference comparison, the `--overwrite` exercise (the refused rerun, the recomputation, the comparison again) and the forge step (writing a tiny ablm-forge checkpoint, scoring it through `evaluate_ablms`, checking it against log-probabilities computed directly).

## Running it

On a laptop, from the directory holding `config.yaml` and the data (the tasks use the GPU when one is visible, and CPU otherwise, many times slower):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/ablm-eval:1.0.4 ablm-eval run config.yaml --cache-dir /work/cache
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/ablm-eval:1.0.4 \
  --container-mounts="$PWD":/work --container-workdir=/work \
  ablm-eval run config.yaml --cache-dir /work/cache
```

The model files land in `--cache-dir` (the default `.ablm-eval-cache` sits in the working directory); the nine registered models are between 30 MB and 2.9 GB each, so keep the cache on scratch and reuse it across runs — a rerun with the same `--cache-dir` downloads nothing.

## Gotchas

- **ablm-forge checkpoints need the forge extra, and the forge that wrote them:** forge registers its `ablm` architecture in process when `ablm` is imported, and its checkpoints carry no `auto_map`, so a forge checkpoint resolves only where ablm-forge is installed — this image installs `ablm-eval[forge]`, and AbLM-Eval's loader imports it; without it the loader stops with an error naming ablm-forge. The extra is unpinned upstream on purpose: the forge that wrote a checkpoint is the one that can load it. This image carries forge `6da77a6` (ablm-forge 0.0.1), and its build fails if another resolves; a checkpoint from a forge whose `ablm` architecture has since changed may not load here, and needs a venv with that forge on top of this image's set. Forge snapshots carry no tokenizer files; AbLM-Eval's loader finds forge's tokenizer through the registered `ablm` type (the test's checkpoint has none, and loads), and a task's `tokenizer_path` overrides it.
- **`--overwrite` reruns:** a task directory that already holds files is refused, and the run stops, unless `--overwrite` is passed — so after a failed or preempted run, rerun the same config with `--overwrite` to reuse the output directory (completed or partial results in it are recomputed, not skipped).
- **`balm-unpaired` is single-chain:** a `paired` task with it in the config is refused by design, naming it. Run it in its own config with `antibody_datatype: unpaired` and single-chain data.
- **The separator comes from the model:** each registered model carries the separator it was trained with and applies it to every task; setting `separator` in a task config alongside such a model is refused. Only full model entries with `separator` of their own, and `balm-unpaired` (which leaves the task's separator alone), differ.
- **`num_proc` defaults to 128** dataset-preparation workers; on a node with fewer cores, set it to what the node has.
- **Rich wraps at 80 columns** when a terminal is not a tty: export `COLUMNS=1000` before `ablm-eval models` in scripts, or the table's repository and revision columns wrap.
- **Classification's `attention_classifier` follows the model:** the task field is unset by default — a model whose config defines it (`balm-moe` only) gets it on, every other model gets nothing, and setting it on a model whose config does not define it is refused with a `ValueError` naming the model. A task with no `training_args` trains on AbLM-Eval's defaults (warmup_steps 0.1, bf16, batch 32, 3 epochs, report_to none); spell out `training_args` only to depart from them. One trap is the caller's, not AbLM-Eval's: under Slurm/pyxis the caller's environment is passed into the container, and an ambient `LOCAL_RANK`/`RANK` with no `WORLD_SIZE` kills a CLI validating a config whose task carries `training_args` — transformers' `TrainingArguments`, built while the config is validated, inits `torch.distributed` at the `env://` rendezvous and dies (`environment variable WORLD_SIZE expected, but not set`); unset them for the CLI, which is a single process that launches its own training through accelerate and sets the training subprocess's variables itself. A task with no `training_args` builds no `TrainingArguments` at validation, so it does not hit the trap. The trap is not gone in the image: the classification check's in-image run arrived with `LOCAL_RANK=0` and `RANK=0` and no `WORLD_SIZE`, and its probe failed exactly there until they were unset.
- **The test** (`test/run.sh`) runs `per_position` through the CLI on AbLM-Eval's own fixture `test/paired-mini.parquet` — the 8 public pairs from Jaffe et al. 2022 (figshare 19617633), heavy chains 124-134 and light 107-113 residues — with the 8M checkpoint `brineylab/8M_full_checkpoint-500000` @ `01e6532f` spelled out as a full model entry with separator `<cls><cls>`, fetched at test time. It compares the pooled mean per-residue log-probability and each sequence's mean with `test/reference.json`, computed once with `evaluate_ablms` (AbLM-Eval's Python API) at commit `641cff8` on the same fixture and model, to 1e-4 — unchanged for 1.0.4: the numpy-2 re-resolve and the AbLM-Eval bump leave it at -0.311014, as before. It checks every registered model's name is listed by `ablm-eval models`, and exercises the documented `--overwrite`: the same config rerun without it is refused (`The directory '...' exists and is not empty!`), and rerun with it the results are recomputed and the comparison passes again. Then `test/forge.py` writes a tiny seeded `AblmForMaskedLM` with `save_pretrained` (no forge checkpoint is public), scores it on the same fixture through `evaluate_ablms`, and checks the token ids and every sequence's mean against the log-probabilities computed directly from the model in memory, to 1e-4.
- **The threshold the fixture crosses:** AbLM-Eval's per-position path has no size switch of its own — no chunking, and no length cutoff unless a task sets `max_len` and `truncate` — so the step that changes with the input is its form. The fixture is paired at full length: both chains joined by the separator, 239 tokens for the first input, so every forward pass is a real masked batch of one row per residue of both chains (235 for the first input) at the length the paired models were trained on, through the separator handling a single short chain never reaches. The forge step crosses the threshold 1.0.4 adds: a checkpoint of `model_type` `ablm` with no `auto_map`, which transformers' Auto classes resolve only through forge's registration. The test does not cover classification, `balm-moe`'s routing or the other registered models; the campaign's all-models and classification checks above do.
- **Licenses:** AbLM-Eval is MIT; every model the registry names (and the 8M test checkpoint) is MIT-licensed on the Hub. The image carries no weights — it downloads them at run time — so a run needs the network and a place to put `--cache-dir`.
