---
name: openfold3
description: "OpenFold3 all-atom structure prediction: an open AlphaFold 3 reproduction that co-folds proteins, DNA, RNA, and ligands as a complex from sequence, single-sequence or with MSAs, and reports pLDDT, pTM, ipTM, and PAE per sample. Reach for it for general-purpose folding of a protein or a complex offline on one GPU of 32 GB or more."
metadata:
  version: 1.0.0
---
# openfold3

Image: `ghcr.io/brineylab/toolshed/openfold3:1.0.0`

OpenFold3 (`openfold3` 0.5.1 on PyPI, from the AlQuraishi Lab and the OpenFold consortium), run through its own `run_openfold predict` program on PyTorch 2.12.1's CUDA 13.0 build (the torch upstream's lockfile resolves) with cuEquivariance 0.10.0. The parameters are 0.5.1's default checkpoint, `openbind-2025-06-30-174k` (`of3-ob-2025-06-30-174k.pt`, the OpenBind-0 weights), from the OpenFold AWS Open Data bucket (`s3://openfold3-data/openfold3-parameters/`), checked against its size and SHA-256 and in the image at `$OPENFOLD_CACHE` (`/opt/openfold3`), where `run_openfold` finds it with no flag; it needs no network.

## Commands

```bash
run_openfold predict --query-json query.json --use-msa-server=False --use-templates=False --num-diffusion-samples 5 --num-model-seeds 1 --output-dir out
```

`--query-json` names the input; `--output-dir` receives the results. `--use-msa-server=False --use-templates=False` is the mode that needs no network: **the MSA server is on by default**, so leaving the flag out sends the query's protein sequences to the public ColabFold server (see Gotchas). `--num-diffusion-samples` (default 5) is structures per seed; `--num-model-seeds` how many seeds. `--runner-yaml` takes a YAML of further settings, among them `output_writer_settings: {structure_format: pdb}` for PDB output and `model_update: {presets: [predict, low_mem]}` for large complexes. `run_openfold predict --help` lists every option.

## Inputs and outputs

- **In:** a JSON object `{"queries": {"<query_id>": {"chains": [...]}}}`. Each chain has `molecule_type` (`protein`, `dna`, `rna`, `ligand`), `chain_ids` (a list; identical chains share one entry), and a `sequence`, or for a ligand `smiles` or `ccd_codes`. A query's `"use_msas": false` runs it single-sequence explicitly; a protein chain's `main_msa_file_paths`/`paired_msa_file_paths` give precomputed MSAs (`.a3m`, `.sto`, `.npz`), files or directories. An MSA file is read only under a name OpenFold3 knows: an A3M from ColabFold is `colabfold_main.a3m` (others are `uniref90_hits`, `mgnify_hits`, `bfd_uniref_hits`, and the rest of `max_seq_counts` in upstream's `dataset_config_components.py`); a homodimer's chains need no paired MSA. Seeds do not go in the JSON. The full schema is upstream's input format reference (openfold-3.readthedocs.io). A homodimer, single-sequence and with an MSA read from a file:

```json
{"queries": {
  "hivpr": {"use_msas": false, "chains": [{"molecule_type": "protein", "chain_ids": ["A", "B"], "sequence": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"}]},
  "hivpr_msa": {"use_msas": true, "chains": [{"molecule_type": "protein", "chain_ids": ["A", "B"], "sequence": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF", "main_msa_file_paths": ["colabfold_main.a3m"]}]}}}
```

- **Out:** under `--output-dir`, `<query_id>/seed_<seed>/` holding per sample `<query_id>_seed_<seed>_sample_<n>_model.cif` (n from 1), `..._confidences.json` (per-atom `plddt`, `pae`, `pde`), and `..._confidences_aggregated.json` with `avg_plddt`, `ptm`, `iptm`, `sample_ranking_score`, `has_clash`, `disorder`, `gpde`, `chain_ptm`, `chain_pair_iptm`, `bespoke_iptm`; plus `timing.json` and the run's resolved configs.

## Compute

One GPU with at least 32 GB (upstream's minimum; most of its testing was on 40 GB A100s) and NVIDIA driver 580 or newer (the image is CUDA 13). On an L40S (48 GB), the test's 198-residue dimer at one sample takes about 12 s of prediction and 29 s for the whole `run_openfold predict` (the checkpoint's load included), at a peak of 2.3 GB of GPU memory; five samples take 14 s of prediction. The test's single-sequence and MSA queries together take 37 s in one `run_openfold predict` (2.4 GB peak); the whole job, the image's import on the node included, took under 6 minutes. Larger complexes were not measured. The record is from an L40S; the CUDA 13 build carries Blackwell's kernels, so a B200 is expected to work but is untested (see Gotchas for the kernel compile).

## Running it

On a laptop, from the directory holding `query.json`:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/openfold3:1.0.0 run_openfold predict --query-json query.json --use-msa-server=False --use-templates=False --num-diffusion-samples 5 --num-model-seeds 1 --output-dir out
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/openfold3:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work run_openfold predict --query-json query.json --use-msa-server=False --use-templates=False --num-diffusion-samples 5 --num-model-seeds 1 --output-dir out
```

## Gotchas

- `--use-msa-server` defaults to on. Without `--use-msa-server=False`, every run sends its protein sequences to the public ColabFold server and fails with no network. Precomputed MSAs need no network: the test reads a 128-row A3M from a file; the server route is not exercised.
- Seeds are derived, not given: `--num-model-seeds N` draws N seeds from a start seed of 42 (a runner YAML's `experiment_settings.seeds` changes it), so `--num-model-seeds 1` always runs seed 2746317213 and a rerun repeats it; outputs land under `seed_<seed>/`.
- Single-sequence mode folds poorly, and its confidence says so. On the test's HIV-1 protease dimer it gives protein-shaped but wrong folds: 15–20 Å CA RMSD from the 1HVR crystal over five samples, avg pLDDT 33–43, ipTM under 0.1. With ColabFold's 5,397-row MSA read from a file, 0.45–1.7 Å (under 1 Å outside the flexible flaps) at avg pLDDT 91–92; with the test's 128 rows, 1.66 Å at 91.6. Read an avg pLDDT near 40 as a wrong fold, not a modest one, and give the model an MSA when accuracy matters. A warning that the Triton evoformer kernel runs "untuned" on a non-HIP backend is about speed on NVIDIA GPUs, not correctness.
- The parameters are OpenBind-0 (`openbind-2025-06-30-174k`), 0.5.1's default; older docs that name `openfold3_p2_v1` or the preview2 weights describe releases before 0.5.
- DeepSpeed's evoformer attention kernel is not installed (it needs CUTLASS); cuEquivariance is the kernel path.
- The test folds HIV-1 protease as a homodimer (2 × 99 = 198 residues), past cuEquivariance's 100-residue crossover (`CUEQ_TRIATTN_FALLBACK_THRESHOLD`), so it exercises the kernel path, once single-sequence and once with a 128-row MSA that must land within 3 Å CA RMSD of the 1HVR crystal. Inputs larger than that were not measured. On boltz-2, the same Triton kernels' first compile on a Blackwell B200 did not finish in 29 minutes; if a B200 run stalls there, raising `CUEQ_TRIATTN_FALLBACK_THRESHOLD` and `CUEQ_TRIMUL_FALLBACK_THRESHOLD` past the input's size (e.g. `export CUEQ_TRIATTN_FALLBACK_THRESHOLD=1000000 CUEQ_TRIMUL_FALLBACK_THRESHOLD=1000000`) runs plain PyTorch and compiles nothing.
- Code and weights are Apache-2.0 (the license is in the image at `/opt/openfold3/LICENSE`).
