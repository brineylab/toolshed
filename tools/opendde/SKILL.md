---
name: opendde
description: "OpenDDE all-atom structure prediction: co-folds proteins, DNA, RNA, ligands, and ions as a complex from sequence, single-sequence or with MSAs, and reports pLDDT, pTM, and ipTM per sample. Reach for it for general-purpose folding of a protein or a complex, such as an antibody's heavy and light chains, offline on one GPU."
metadata:
  version: 1.0.0
---
# opendde

Image: `ghcr.io/brineylab/toolshed/opendde:1.0.0`

OpenDDE (`opendde` 1.2.0 on PyPI, from Aureka Research; an AlphaFold 3-style all-atom co-folding model, `opendde_v1`, 656M parameters, training data to 2021-09-30; technical report arxiv.org/abs/2607.03787), run through its own `opendde pred` program on PyTorch 2.7.1's CUDA 12.8 build with cuEquivariance 0.10.0 and Triton 3.3.1. The general-purpose checkpoint `opendde.pt` and the CCD files OpenDDE reads (`components.cif` and its RDKit pickle, and the template metadata) come from Hugging Face's `aurekaresearch/OpenDDE` at revision `eddd563ce96571f784012edd8f045181c8f8627d`, checked against the sizes and SHA-256s opendde 1.2.0 verifies, and are in the image at `$OPENDDE_ROOT_DIR` (`/opt/opendde`); it needs no network.

## Commands

```bash
opendde pred -i input.json -o out -n opendde_v1 --use_msa false --use_template false --use_rna_msa false --seeds 0 --sample 5
```

`-i` is the input JSON; `-o` the output directory. `--use_msa false --use_template false --use_rna_msa false` is single-sequence mode, the mode that needs no network and no search databases; leave any of them out and OpenDDE tries to search (see Gotchas). `--seeds` takes a comma-separated list of seeds (overriding the JSON's `modelSeeds`); `--sample` (`-e`, default 5) is how many diffusion samples per seed; `--cycle` (`-c`) recycling steps and `--step` (`-p`) diffusion steps trade speed for quality; `--dtype {bf16,fp32}`; `--trimul_kernel`/`--triatt_kernel {auto,cuequivariance,torch}` choose the triangle kernels (`auto` is cuEquivariance). `opendde pred --help` lists every option; `opendde doctor` prints the environment.

## Inputs and outputs

- **In:** a JSON list of jobs in AlphaFold Server's dialect. Each job has a `name` (used in output paths), `sequences` (a list of entities, each with exactly one key: `proteinChain`, `dnaSequence`, `rnaSequence`, `ligand`, or `ion`), and optionally `modelSeeds` and `covalent_bonds`. Every entity has `count`; an optional `id` lists its chain IDs (as long as `count`). A protein entity can carry `pairedMsaPath` and `unpairedMsaPath` (A3M files) and `modifications` (`{"ptmType": "CCD_MSE", "ptmPosition": 1}`, 1-based). Ligands are `{"ligand": {"ligand": "CCD_ATP", "count": 1}}` or a SMILES. The full schema is upstream's `docs/infer_json_format.md` (github.com/aurekaresearch/OpenDDE). A homodimer:

```json
[{"name": "hivpr", "modelSeeds": [0],
  "sequences": [{"proteinChain": {"sequence": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF", "count": 2, "id": ["A", "B"]}}]}]
```

- **Out:** under `-o`, a `<name>/seed_<seed>/predictions/` directory per job and seed holding, per sample ranked by `ranking_score` (`sample_0` the best), `<name>_sample_<i>.cif` (atom pLDDT in the B-factor column, 0–100) and `<name>_summary_confidence_sample_<i>.json` with `plddt` (mean, 0–100), `ptm`, `iptm` (0–1), `ranking_score`, `has_clash`, `gpde`, and per-chain and chain-pair pTM/ipTM.

## Compute

One GPU. Compute numbers come from this tool's test record (filled in below when measured).

## Running it

On a laptop, from the directory holding `input.json`:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/opendde:1.0.0 opendde pred -i input.json -o out -n opendde_v1 --use_msa false --use_template false --use_rna_msa false --seeds 0 --sample 5
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/opendde:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work opendde pred -i input.json -o out -n opendde_v1 --use_msa false --use_template false --use_rna_msa false --seeds 0 --sample 5
```

## Gotchas

- Single-sequence mode costs accuracy, more for complexes whose interfaces an MSA would inform. With `--use_msa true` (the default) and no A3M paths in the JSON, OpenDDE asks the public ColabFold MMseqs2 server for MSAs, which needs the network; A3M files named in the JSON need none. Neither route is exercised by this tool's test. Templates and RNA MSAs need search databases and HMMER, which this image does not carry.
- Upstream calls OpenDDE a preview and says checkpoints may change between versions; this image pins revision `eddd563c…` of the weights.
- Only the general-purpose checkpoint is carried; upstream's antibody–antigen checkpoint (`opendde_abag.pt`) is not.
- The test folds HIV-1 protease as a homodimer (2 × 99 = 198 residues), past cuEquivariance's 100-residue crossover, so it exercises the Triton kernels and their first-use compile (gcc is in the image for it). Inputs larger than that were not measured.
- Code and weights are Apache-2.0 (the license is in the image at `/opt/opendde/LICENSE`).
