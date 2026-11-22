---
name: esmfold2-fast
description: "ESMFold2-Fast all-atom structure prediction: Biohub's single-sequence ESMFold2, on the ESM C 6B language model, folds proteins, DNA, RNA, and ligands as a complex with no MSA and reports pLDDT, pTM, ipTM, and PAE. Reach for it to fold a protein or a complex fast from sequence alone, offline on one GPU; use esmfold2 when MSAs are at hand."
metadata:
  version: 1.0.0
---
# esmfold2-fast

Image: `ghcr.io/brineylab/toolshed/esmfold2-fast:1.0.0`

ESMFold2-Fast (Hugging Face `biohub/ESMFold2-Fast` at revision `45fe8656f5b3ef493c17fcf9abe9a2968902e712`; Biohub, Candido et al. 2026, bioRxiv 10.64898/2026.06.03.729735), the single-sequence ESMFold2: a folding trunk and diffusion structure module on the ESM C 6B language model, which it bundles. Run through the `esm` package 3.5.0 on PyTorch 2.14.1's CUDA 13.0 build by `esmfold2`, a short program in this image that reads upstream's own input JSON. The weights are in the image at `$ESMFOLD2_MODEL` (`/opt/model`) with the ESM C backbone (`esmc.*` tensors) stored in bf16 and the folding layers in fp32 as published, and the CCD pickle ligands and modified residues need (`ccd.pkl`, from `biohub/ESMFold2` at `69869f737beffec5294845ede23db5fc0b4f509e`, the Fast repository carries none) at `$ESMCFOLD_CCD_PATH`; it needs no network.

## Commands

```bash
esmfold2 predict --in jobs.json --out out --seed 0
```

`--in` is a JSON file of one job or a list of them; `--out` is created if missing, and a rerun into it overwrites the same files. `--samples` is diffusion samples per job (default 1); `--seed` (default 0) seeds input preparation and sampling; `--loops` (trunk recycling, default 3) and `--steps` (diffusion steps, default 50) trade speed for quality, defaults from the model card's Fast example (`$ESMFOLD2_LOOPS`, `$ESMFOLD2_STEPS`). The model loads once per invocation, so put a batch in one file. Every job is checked before the model loads; a malformed one stops the run with exit 1 and a line naming it.

## Inputs and outputs

- **In:** each job is `{"name": ..., "sequences": [...]}`, optionally with `covalent_bonds` and `pocket`: upstream's serialized `StructurePredictionInput`, read by `esm`'s own `deserialize_structure_prediction_input`. A sequence entry has a `type` (`protein`, `dna`, `rna`, `ligand`) and an `id`, a chain ID or a list of them for identical copies. Polymers carry a `sequence` and optional `modifications` (`[{"position": 0, "ccd": "SEP"}]`, 0-based); ligands carry `ccd` (a list of codes) or `smiles`. `name` must be unique in the file and holds no `/`. A homodimer and a protein–ligand pair:

```json
[{"name": "hivpr", "sequences": [{"type": "protein", "id": ["A", "B"], "sequence": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"}]},
 {"name": "kinase_atp", "sequences": [{"type": "protein", "id": "A", "sequence": "MSEQ..."}, {"type": "ligand", "id": "L", "ccd": ["ATP"]}]}]
```

- **Out:** per job and sample `i` (from 0), `<name>_sample_<i>.cif` (all-atom mmCIF), `<name>_sample_<i>.json` with `plddt` (mean over tokens), `ptm`, `iptm` (null for a single chain), and `pair_chains_iptm` (chain × chain), and `<name>_sample_<i>.npz` with per-token `plddt` and the `pae` matrix.

## Compute

One GPU. On an L40S (48 GB), one `esmfold2 predict` of the test's file (the 198-residue HIV-1 protease dimer and the 33-residue GCN4-p1, one sample each at the default 3 loops and 50 steps) takes about 30 s, the 13 GB model's load most of it, at a peak of 8.5 GB of GPU memory. The image is 21 GB: the test job spent about 10 minutes before its first line, most of it the node importing the image. The fp8 backbone needs compute capability 8.9 or newer (L40S, H100, H200, B200) for its speed; an A100 upcasts each matrix multiply. Larger inputs were not measured.

## Running it

On a laptop, from the directory holding `jobs.json`:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/esmfold2-fast:1.0.0 esmfold2 predict --in jobs.json --out out --seed 0
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/esmfold2-fast:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work esmfold2 predict --in jobs.json --out out --seed 0
```

## Gotchas

- This is the single-sequence model: it has no MSA encoder, so an `msa` in the input does nothing. For MSA conditioning use the `esmfold2` tool, the same program on the full model.
- The ESM C backbone is stored in bf16, halving the image: `esm` 3.5.0 casts the published fp32 backbone to bf16 at load before converting its projections to fp8 (`set_esmc_precision`; fp8 matrix multiplies on compute capability 8.9 and newer, upcast on older GPUs), so the loaded model is the one `esm` builds from the fp32 originals. The folding layers are fp32 as Biohub published them.
- Two warnings print at every load and are expected: `flash-attn` is not installed, so ESM C's attention runs in plain PyTorch (Biohub's wheel is built for torch 2.11 and compute capability 8.0–9.0; this image is torch 2.14.1), and "fp8 ESMC on cpu: no fp8 GEMM here" comes from converting the backbone on the CPU before it moves to the GPU, where compute capability 8.9 and newer runs the fp8 GEMMs.
- The test folds HIV-1 protease as a homodimer (2 × 99 = 198 residues) and GCN4-p1 (33 residues) from one file. Ligands, nucleic acids, modifications, covalent bonds, and pockets are upstream's schema, documented here but not exercised by the test. Inputs larger than the fixture were not measured.
- Code and weights are MIT (Biohub's `LICENSE.md` and `THIRD_PARTY_NOTICE.md` are in the image beside the weights). Biohub's acceptable use policy (biohub.org/acceptable-use-policy) applies to the model.
