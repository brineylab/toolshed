---
name: ssrmsd
description: "Self-consistency RMSD (ssRMSD) scoring for predicted antibody-antigen and protein complex structure ensembles: scores single ensembles or parallel batches across heterogeneous structure predictors (AlphaFold 3, Boltz-2, ESMFold2, OpenFold3, OpenDDE). Reach for it to measure or compare structural consistency across samples from one or more structure predictors."
metadata:
  version: 1.0.0
---
# ssrmsd

Image: `ghcr.io/brineylab/toolshed/ssrmsd:1.0.0`

`ssRMSD` 0.1.1 (github.com/brineylab/ssrmsd, MIT) on Python 3.12 with `gemmi` 0.7.5, `numpy` 2.5.5, and `typer` 0.29.2. It computes pairwise Cα Kabsch RMSD statistics across predicted structure ensembles — either within a single predictor or combined across heterogeneous predictors (`.cif`, `.mmcif`, `.pdb`, `.ent`, `.gz`, `.tar.gz`/`.tgz`/`.tar`, and `.zip`, read directly in memory). Chains are matched across models by sequence rather than chain ID or file order, with optional separate superposition (`--align-chains`) and scoring (`--score-chains`) subsets. CPU only, no model weights, no network.

## Commands

Score a batch of ensembles from one or more predictor output roots (the primary pipeline command):

```bash
ssrmsd batch --dir af3=/work/af3_out --workers 8 -o /work/scores.csv
```

Cross-join two or more predictors on matching target IDs, superposing on antigen chain `A` (as named by `af3`) and measuring RMSD on antibody chains `B,C`, while emitting both summary and pairwise tables:

```bash
ssrmsd batch \
  --dir af3=/work/af3_out \
  --dir esmfold2=/work/esm_out \
  --join intersection \
  --align-chains A --score-chains B,C --ref-source af3 \
  --workers 8 \
  --pairwise-out /work/pairs.csv \
  -o /work/scores.csv
```

How each structure predictor in this toolshed plugs into `ssrmsd batch --dir`:

| Toolshed predictor | Output layout | `--dir` path to pass |
| --- | --- | --- |
| `alphafold3-blackwell` | `<out>/<target>/seed-<s>_sample-<i>/*.cif` (+ top-level duplicate) | `--dir af3=<out>` (top-level duplicate skipped automatically) |
| `openfold3` | `<out>/<target>/seed_<seed>/<target>_seed_<seed>_sample_<n>_model.cif` | `--dir openfold3=<out>` |
| `opendde` | `<out>/<target>/seed_<seed>/predictions/<target>_sample_<i>.cif` | `--dir opendde=<out>` |
| `boltz-2` (batch dir) | `<out>/boltz_results_<batch>/predictions/<target>/<target>_model_<i>.cif` | `--dir boltz=<out>/boltz_results_<batch>/predictions` |
| `esmfold2` / `esmfold2-fast` | `<out>/<target>_sample_<i>.cif` (flat in `<out>`) | `--dir esmfold2=<out>` (flat `*_sample_<k>` files grouped by `<target>`) |

Score an arbitrary batch from a CSV or TSV manifest (`ensemble_id,path[,source]`):

```bash
ssrmsd batch --manifest /work/ensembles.csv --format jsonl --workers 8 -o /work/scores.jsonl
```

Score a single ensemble from files, directories, or archives (optionally prefixed with `label=`):

```bash
ssrmsd score /work/af3_out/mab1 -o /work/mab1.csv
ssrmsd score af3=/work/mab1_af3.tar.gz esmfold=/work/mab1_esm/ \
  --id mab1 --align-chains A --score-chains B,C --ref-source af3 \
  --pairwise-out /work/mab1_pairs.csv -o /work/mab1.csv
```

Check the installed package version:

```bash
ssrmsd --version
```

## Inputs and outputs

- **In:** structure files (`.cif`, `.mmcif`, `.pdb`, `.ent`, plain or `.gz`), directories, or `.tar.gz`/`.tgz`/`.tar`/`.zip` archives (read in memory without extracting to disk). `--dir [label=]ROOT` (repeatable) discovers direct child directories, archives, or flat `<target>_{model,sample}_<k>.*` files under each `ROOT` and merges matching `ensemble_id`s using `--join union` (default) or `--join intersection`. `--manifest PATH` (mutually exclusive with `--dir`) reads CSV/TSV rows with columns `ensemble_id`, `path`, and optional `source`; multiple rows sharing an `ensemble_id` combine into one ensemble.
- **Out:** CSV (default), TSV, or JSONL (`--format {csv,tsv,jsonl}`) written to `-o/--output` (or stdout). Columns: `ensemble_id`, `n_structures`, `n_ca_align`, `n_ca_score`, `mean_sq_rmsd` ($\frac{1}{\binom{n}{2}}\sum_{i<j} d_{ij}^2$, Å²), `rms_rmsd` ($\sqrt{\text{mean\_sq\_rmsd}}$, Å), `mean_rmsd`, `median_rmsd`, `rmsd_sd` (upper-triangle population SD), legacy parity columns `ssRMSD` ($2\sum_{i<j} d_{ij}^2$) and `SD` (full $n \times n$ matrix SD including zeros), multi-source `within_source_mean_rmsd` and `cross_source_mean_rmsd`, `sources` (e.g. `af3:5,esmfold2:5`), `status` (`ok` or `error`), and `error`. `--pairwise-out PATH` writes one row per unordered pair (`ensemble_id, struct_i, struct_j, source_i, source_j, rmsd`).

## Compute

CPU only, small memory footprint (~235 MB image; tens of MB RAM per worker), scaling across ensembles with `--workers` (defaults to host `os.cpu_count()` in `batch`; under Slurm cgroups always pass `--workers` matching `--cpus-per-task`, see Gotchas). Measured inside the `linux/amd64` container on synthetic 490-residue 3-chain antibody–antigen complexes (250 + 125 + 115 residues, 5 mmCIF samples per ensemble): one ensemble (`ssrmsd score`) finishes in 0.17 s (mostly Python CLI startup); a batch of 100 ensembles finishes in 1.48 s at `--workers 1` (~67 ensembles/s) and 0.38 s at `--workers 8` (~260 ensembles/s, `ProcessPoolExecutor` startup included). Real all-atom mmCIF files are larger to parse than Cα-only synthetic fixtures, so budget modestly more I/O time on full-atom predictions; memory stays flat in the number of ensembles. The tool's full test (`test/run.sh`) finishes in ~1.3 s.

## Running it

On a laptop, from the directory holding the predictor outputs:

```bash
podman run --rm -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/ssrmsd:1.0.0 \
  ssrmsd batch --dir af3=/work/af3_out --dir esmfold2=/work/esm_out --join intersection --workers 4 -o /work/scores.csv
```

On a Slurm cluster with pyxis (CPU partition):

```bash
srun --cpus-per-task=8 --mem=8G --container-image=ghcr.io#brineylab/toolshed/ssrmsd:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work \
  ssrmsd batch --dir af3=/work/af3_out --dir esmfold2=/work/esm_out --join intersection --workers 8 -o /work/scores.csv
```

## Gotchas

- **Pass `--workers` explicitly on Slurm:** `ssrmsd batch` defaults `--workers` to `os.cpu_count()` (the node's total host CPUs, not the Slurm cgroup allocation). On a shared high-core node, omitting `--workers` can spawn far more processes than `--cpus-per-task` / `--mem` budgeted; always pass `--workers` matching `--cpus-per-task`.
- **Boltz-2 batch directory path:** `boltz predict` nests targets under `<out_dir>/boltz_results_<batch>/predictions/<target>/`. Pass `.../predictions` to `--dir`, not `<out_dir>` itself.
- **AF3 top-level duplicate exclusion:** When a directory or archive contains `seed-*_sample-*` subdirectories, top-level `*_model.cif` duplicates are skipped automatically so the top-ranked sample is not double-counted; override with `--include-top-level` or `--glob`.
- **Reference chain IDs (`--align-chains`, `--score-chains`, `--ref-source`):** Chain IDs refer to the reference structure and are propagated to every other structure by sequence alignment. When mixing predictors that assign different chain IDs, pass `--ref-source <label>` so the reference is explicit; if `--ref-source` is absent from an ensemble, that ensemble fails with `status=error` rather than falling back silently.
- **Homomers and identical chains are refused in v1:** If two required chains have sequence identity `>= --min-chain-id` (default `0.90`), `ssrmsd` fails that ensemble loudly (`status=error`) instead of risking a swapped chain assignment. Homodimers such as the HIV-1 protease fixtures in `esmfold2`, `openfold3`, and `opendde` will therefore record `status=error`.
- **Poly-G linkers and coverage guards:** Single-chain multimers joined by poly-G runs (`>= --linker-min-len`, default `10`) are split automatically when a structure has fewer chains than `max_chains`. Pairwise RMSD uses the ensemble-wide Cα intersection, and every required chain must retain `>= --min-coverage` (default `0.80`) relative to the longest copy across the ensemble.
- **Exit codes in `batch` vs `score`:** `ssrmsd score` exits `1` when the ensemble records `status=error`. `ssrmsd batch` writes every row (`ok` and `error`) and exits `0` so isolated failed predictions do not abort a Slurm array or campaign, unless `--fail-fast` is passed or a fatal input/manifest error occurs.
- **What the test covers and the threshold it crosses:** `test/run.sh` runs `ssrmsd batch --dir` on 3 synthetic 3-chain antibody–antigen ensembles with `--workers 2`, crossing `score_batch`'s multiprocessing threshold (`workers > 1` and `len(ensembles) > 1` switches from sequential execution to `ProcessPoolExecutor`). It verifies AF3 decoy duplicate exclusion, flat `esmfold2` `*_sample_*.cif` grouping and `.json`/`.npz` sidecar ignoring, permuted chain ID matching, `.tar.gz` + poly-G-linked `.pdb` scoring via `ssrmsd score`, manifest JSONL error isolation with a homomer fixture, Pyxis `python3 -I` isolation against a poisoned `PYTHONPATH` from a read-only working directory, and analytical RMSDs within `1e-4` on mmCIF inputs (`1e-4` relative / `1e-3` absolute on 3-decimal PDB inputs). Very large batches (10,000+ ensembles) and full-atom crystal structures were not measured in the test.
- **License:** `ssRMSD` is MIT-licensed (`/opt/ssrmsd/LICENSE` in the image). No model weights are carried.
