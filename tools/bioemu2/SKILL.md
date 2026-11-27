---
name: bioemu2
description: "BioEmu2 (microsoft/bioemu2): a ~500M-parameter diffusion model that predicts folded structures, samples equilibrium all-atom ensembles and propagates structures forward in time from the bioemu2 CLI. Reach for it for conformational heterogeneity and time-lagged dynamics of proteins, complexes, nucleic acids and ligands at a fraction of MD's cost; use boltz-2 for plain structure+confidence and openmm for explicit-solvent MD."
metadata:
  version: 1.0.0
---
# bioemu2

Image: `ghcr.io/brineylab/toolshed/bioemu2:1.0.0`

Microsoft Research AI for Science's BioEmu2 public preview (MIT), upstream release 0.1.0rc0 with the `cuda` extra (cuEquivariance 0.11.1, CUDA 13), run through the upstream `bioemu2` command line on PyTorch 2.13 / Python 3.12, the interpreter upstream benchmarked. It folds one structure, samples independent equilibrium all-atom frames and samples endpoints a chosen time lag after a starting conformation — independent samples, not a molecular-dynamics trajectory. One GPU only: multi-GPU context-parallel sampling (`bioemu2 cp`) is out of this tool's scope.

The checkpoint `bioemu2-preview` from `microsoft/bioemu2` revision `dd4b0c1b69984b68edcb6200a544213be4d75242` (~2 GB, Hub `checkpoints/bioemu2-preview/checkpoint.ckpt` + `config.yaml`) is **in the image** at `HF_HUB_CACHE=/toolshed/weights`, in the standard `models--microsoft--bioemu2/snapshots/<commit>/` cache layout, so upstream's default named-checkpoint lookup never downloads anything; the local alternative is `--checkpoint-path FILE.ckpt` with its `config.yaml` beside it.

## Commands

Predict a folded structure from a PDB ID (downloads biological assembly 1 from RCSB, non-water ligands included) or from your own system YAML:

```bash
bioemu2 predict-structure --pdb-id 1UBQ --num-samples 1 --out outputs/ubiquitin
bioemu2 predict-structure --system system.yaml --num-samples 1 --out outputs/ubiquitin
```

Sample an equilibrium ensemble (the frames are independent, not a time series):

```bash
bioemu2 sample-equilibrium --pdb-id 1AKE --num-samples 8 --out outputs/adk
bioemu2 sample-equilibrium --system /opt/bioemu2/examples/chignolin.yaml --num-samples 8 --out outputs/chignolin
```

Propagate sampled structures forward in time, one endpoint per input frame:

```bash
bioemu2 propagate --x0-cif outputs/adk/topology.cif --x0-xtc outputs/adk/samples.xtc \
  --lag-time-ns 1 --num-steps 200 --out outputs/adk-1ns
```

Or draw several endpoints from one starting CIF instead (`--num-samples 8` in place of `--x0-xtc`). Upstream recommends lags of 1–1000 ns; endpoints are aligned to their own starting frame. All three commands accept `--system YAML` or `--pdb-id ID`; `propagate` may omit both and infer the system from `--x0-cif` (valid for BioEmu2's own sampling outputs, which carry entity metadata).

Every command writes, under `--out DIR`: `topology.cif` + `samples.xtc` (the topology is frame 0; propagate's XTC holds endpoints only, no starting frames), `system.yaml`, `command.txt` and `physicality.json`, one entry per frame. Controls: `--num-samples`, `--batch-size`, `--seed`, `--num-steps` (200 by default upstream), `--sampler edm|edm_gd|edm_fk|edm_fkc`, `--precision bf16-mixed|32-true`, `--matmul-precision`, `--compile-denoiser`, `--physicality-filter`, `--checkpoint`, `--checkpoint-path`, `--checkpoint-cache-dir`. `bioemu2 --help` and `bioemu2 COMMAND --help` show everything; `bioemu2 convert-boltz INPUT.yaml --out system.yaml` converts a Boltz-1/2 input to system YAML.

## Inputs and outputs

**System YAML** is upstream's schema (`entities`, each `entity_type: protein|rna|dna|ligand|branched` with `sequence`/`asym_ids`, ligands by `ccd_code`, `smiles` or `inchi`), e.g. an antibody–antigen complex or a protein with a small molecule. Five upstream examples are in the image at `/opt/bioemu2/examples/` (`ake.yaml`, `chignolin.yaml`, `meteorin_like_7tlw.yaml`, `myoglobin_101m.yaml`, `shp2_3tkz.yaml`); multi-chain systems list one entity per unique sequence with one or more `asym_ids`.

**MSA.** By default upstream `get_msa` queries the public ColabFold server for each unique protein sequence (paired alignments too for heteromers) and caches the raw A3M response in `BIOEMU2_MSA_CACHE`. The image pre-caches the alignments for this skill's documented fixtures (1AKE, 1UBQ, chignolin and the packaged ake.yaml), so those exact commands run with no network. For other sequences a fresh ColabFold search needs network; set `BIOEMU2_MSA_CACHE` to a writable directory to retain it between jobs (the image's default cache is read-only). Ligands and nucleic acids need no MSA entries.

**Your own A3M** (upstream's CLI has no switch for this; the image's wrapper adds it, calling upstream's `get_msa(source=...)`):

```bash
bioemu2 sample-equilibrium --system system.yaml --msa alignments/my-protein.a3m \
  --num-samples 8 --out outputs/ensemble
```

For a heteromer, repeat `--msa-chain CHAIN=FILE.a3m`, giving one file per protein chain. With one unique sequence (monomer or homomer) `--msa` covers all its chains. Each A3M's **first record must be the query sequence of that chain**; upstream validates it and inserts no ColabFold pairing for local files. Single-sequence input is just the query as the only record.

**Outputs:** `samples.xtc` loads with the sibling `topology.cif` (`mdtraj.load('samples.xtc', top='topology.cif')`); every atom of the system is present, heavy-atom charges recorded in the CIF's `_atom_site.pdbx_formal_charge`. Rejected frames stay in the XTC; `physicality.json`'s `frame` indexes tell which passed (`--physicality-filter no_filter` records `not_checked`).

## Compute

One NVIDIA GPU, compute capability 7.5 or newer for this CUDA 13 build (T4/A100/A6000/L40S/H100/B200 class; Volta V100 needs upstream's separate `cuda12` extra, which this image does not carry). The checkpoint loads in about a minute; the test's 20-step chignolin and 214-residue AKE commands each complete in minutes. Peak memory grows with token count N (~N² pair representations, trunk O(N³)); denoising cost is `--batch-size × N²` per step × `--num-steps`. Reduce `--batch-size` (default: all samples) if a large system runs out of memory; BioEmu2 resumes interrupted sampling runs from `<out>/partial` when you rerun the identical command, which `--seed` makes reproducible.

## Running it

On a laptop, from the directory holding your YAML/A3M:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/bioemu2:1.0.0 \
  bioemu2 sample-equilibrium --system system.yaml --msa protein.a3m --num-samples 8 --out out
```

On a Slurm cluster with pyxis (the same three commands as above where the mounts hold the input):

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/bioemu2:1.0.0 \
  --container-mounts="$PWD":/work,/data/my-campaign:/cache \
  --container-workdir=/work --container-env=BIOEMU2_MSA_CACHE=/cache/msa \
  bioemu2 predict-structure --pdb-id 1UBQ --num-samples 1 --out out
```

## Gotchas

- One GPU per run by design; the multi-GPU `bioemu2 cp` subcommand is present upstream but untested here and needs a square mesh of ≥4 GPUs.
- `--pdb-id` needs an entry in the image's RCSB cache (currently 1AKE and 1UBQ) *or* network access to files.rcsb.org; the wrapper upstream-caches those two assemblies so this skill's examples work offline. Arbitrary PDB IDs download live, as upstream does.
- ColabFold's public server is the default MSA source and needs network on a cache miss; prefer your own A3M (`--msa`, `--msa-chain`) or a named writable `BIOEMU2_MSA_CACHE` for reproducibility and for bulk jobs (ColabFold asks that queries be serial from one IP).
- A system YAML with SMILES/InChI ligands or unusual CCD codes fetches CCD component files at run time into `BIOEMU2_CCD_CACHE` (only AP5 is pre-cached, for 1AKE); truly offline runs of novel ligand systems need those files pre-seeded or network.
- `--num-steps 20` is what this tool's own test does for its 30-minute budget; production commands should keep upstream's default 200 (or 500 for guided samplers). The test checks output shape, system composition and offline data/weights plumbing — Chignolin crosses no size-related kernel crossover upstream documents, so large-system triton paths are not covered by the fixture; AKE (214 residues, homodimer + AP5 ligand) covers multi-chain and ligand paths.
- Propagation endpoints are independent samples at the chosen lag, not a time-ordered trajectory; `propagate` refuses to overwrite existing `samples.xtc` in `--out`.
- Weights and code are MIT-licensed: upstream's `LICENSE`/`NOTICES.md` and third-party notices are at `/opt/bioemu2/licenses/`, the model card at `hub-MODEL_CARD.md` beside them.
- Keep `HF_HUB_OFFLINE=1` (the image's default): any named checkpoint other than the shipped `bioemu2-preview` fails offline as intended.
