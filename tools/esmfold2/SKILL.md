---
name: esmfold2
description: "ESMFold2 all-atom structure prediction: Biohub's ESMFold2, on the ESM C 6B language model, folds proteins, DNA, RNA, and ligands as a complex from sequence alone or conditioned on MSAs, and reports pLDDT, pTM, ipTM, and PAE. Reach for it for the most accurate ESMFold2 prediction of a protein or a complex, with or without an MSA, offline on one GPU; esmfold2-fast is the faster single-sequence model."
metadata:
  version: 1.0.0
---
# esmfold2

Image: `ghcr.io/brineylab/toolshed/esmfold2:1.0.0`

ESMFold2 (Hugging Face `biohub/ESMFold2` at revision `69869f737beffec5294845ede23db5fc0b4f509e`; Biohub, Candido et al. 2026, bioRxiv 10.64898/2026.06.03.729735): an MSA encoder, a folding trunk, and a diffusion structure module on the ESM C 6B language model, which it bundles. Run through the `esm` package 3.5.0 on PyTorch 2.14.1's CUDA 13.0 build by `esmfold2`, a short program in this image that reads upstream's own input JSON. The weights are in the image at `$ESMFOLD2_MODEL` (`/opt/model`) with the ESM C backbone (`esmc.*` tensors) stored in bf16 and the folding layers in fp32 as published, and the CCD pickle ligands and modified residues need (`ccd.pkl`, from the same revision) at `$ESMCFOLD_CCD_PATH`; it needs no network.

## Commands

```bash
esmfold2 predict --in jobs.json --out out --seed 0
```

`--in` is a JSON file of one job or a list of them; `--out` is created if missing, and a rerun into it overwrites the same files. `--samples` is diffusion samples per job (default 1); `--seed` (default 0) seeds input preparation and sampling; `--loops` (trunk recycling, default 20) and `--steps` (diffusion steps, default 200) trade speed for quality, defaults from `esm`'s own `fold()` (`$ESMFOLD2_LOOPS`, `$ESMFOLD2_STEPS`); the model card's quick example uses 3 and 50. The model loads once per invocation, so put a batch in one file. Every job is checked before the model loads; a malformed one stops the run with exit 2 and a line naming it.

## Inputs and outputs

- **In:** each job is `{"name": ..., "sequences": [...]}`, optionally with `covalent_bonds` and `pocket`: upstream's serialized `StructurePredictionInput`, read by `esm`'s own `deserialize_structure_prediction_input`. A sequence entry has a `type` (`protein`, `dna`, `rna`, `ligand`) and an `id`, a chain ID or a list of them for identical copies. Polymers carry a `sequence` and optional `modifications` (`[{"position": 0, "ccd": "SEP"}]`, 0-based); ligands carry `ccd` (a list of codes) or `smiles`. A protein or RNA entry may carry an MSA as `"msa": {"sequences": [...]}`: the query first, then aligned rows of the same length (uppercase residues and `-` gaps; strip an A3M's lowercase insertions first, since rows go to `esm` as given). MSAs are made elsewhere; this image searches nothing. `name` must be unique in the file and holds no `/`. A homodimer and a protein–ligand pair:

```json
[{"name": "hivpr", "sequences": [{"type": "protein", "id": ["A", "B"], "sequence": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"}]},
 {"name": "kinase_atp", "sequences": [{"type": "protein", "id": "A", "sequence": "MSEQ..."}, {"type": "ligand", "id": "L", "ccd": ["ATP"]}]}]
```

- **Out:** per job and sample `i` (from 0), `<name>_sample_<i>.cif` (all-atom mmCIF), `<name>_sample_<i>.json` with `plddt` (mean over tokens), `ptm`, `iptm` (null for a single chain), and `pair_chains_iptm` (chain × chain), and `<name>_sample_<i>.npz` with per-token `plddt` and the `pae` matrix.

## Compute

One GPU. Compute numbers come from this tool's test record (filled in below when measured).

## Running it

On a laptop, from the directory holding `jobs.json`:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/esmfold2:1.0.0 esmfold2 predict --in jobs.json --out out --seed 0
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/esmfold2:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work esmfold2 predict --in jobs.json --out out --seed 0
```

## Gotchas

- Without an `msa` a chain folds single-sequence, at some cost in accuracy; `esmfold2-fast` is the model built for single-sequence folding, with the same program and input.
- The ESM C backbone is stored in bf16, halving the image: `esm` 3.5.0 casts the published fp32 backbone to bf16 at load before converting its projections to fp8 (`set_esmc_precision`; fp8 matrix multiplies on compute capability 8.9 and newer, upcast on older GPUs), so the loaded model is the one `esm` builds from the fp32 originals. The folding layers are fp32 as Biohub published them.
- The test folds HIV-1 protease as a homodimer (2 × 99 = 198 residues) alone and with a five-row MSA, and GCN4-p1 (33 residues), from one file. Ligands, nucleic acids, modifications, covalent bonds, and pockets are upstream's schema, documented here but not exercised by the test. Inputs larger than the fixture were not measured.
- Code and weights are MIT (Biohub's `LICENSE.md` and `THIRD_PARTY_NOTICE.md` are in the image beside the weights). Biohub's acceptable use policy (biohub.org/acceptable-use-policy) applies to the model.
