# Structure predictors: opendde, openfold3, esmfold2-fast, esmfold2

Status: draft 2026-11-21, for Bryan's approval.

## Intent

The lab wants general-purpose structure prediction from several current models, called the same way as the shed's other tools, so a question can fold a protein or a complex with whichever model suits it and compare them. Four tools, each its own pull request, in this order:

1. `opendde` — OpenDDE, general-purpose checkpoint.
2. `openfold3` — OpenFold3, default (OpenBind) checkpoint.
3. `esmfold2-fast` — ESMFold2-Fast, single-sequence.
4. `esmfold2` — ESMFold2, single-sequence or MSA-conditioned.

`opendde` first because it is the simplest (upstream CLI, one 2.6 GB checkpoint, documented offline mode). `esmfold2-fast` before `esmfold2` because it settles the wrapper and the sharded-weights build on the smaller of the two problems (no MSA path).

## Goals, non-goals, anti-goals

- **Goals:** each tool folds a multi-chain protein complex from sequence with no network, on one GPU, and writes mmCIF plus its model's confidence metrics; its skill tells an agent how to call it, including the inputs upstream supports beyond protein (ligands, DNA/RNA) and how to supply MSAs.
- **Non-goals:** OpenDDE's antibody–antigen checkpoint (`opendde_abag.pt`); MSA generation inside any image (MSA search databases are the data companion's, DESIGN §1); multi-GPU (Fold-CP); a common input or output format across the four tools.
- **Anti-goals:** anything that downloads at run time (weights, CCD, model code); a wrapper that invents an input format where upstream has one; a test that only exercises a path below a code-path threshold (CONTRIBUTING, "The test").

## Constraints (from the shed, binding)

AGENTS.md's rules and CONTRIBUTING.md apply as written: weights in the image, environment first and weights last, `COPY test /toolshed/test` last, every package and base pinned, `HF_HUB_OFFLINE=1`, `NVIDIA_*` set on a glibc base, Python wrappers under `python3 -I`, every skill example a full command line, test on a kilobyte fixture with a known answer under `/tmp` in one GPU and 30 minutes with `--network none`. All four upstream licenses permit publishing: OpenDDE Apache-2.0 (code and weights), OpenFold3 Apache-2.0 (code and weights; the Hugging Face copy is gated, the AWS copy is not), ESMFold2 and ESMFold2-Fast MIT (code and weights, not gated). Each image carries the upstream license files beside the weights.

This machine has no GPU: builds and pushes happen here; each test runs on the cluster through `scripts/test <name> srun --gres=gpu:1 ...` (an L40S, 48 GB, as boltz-2's did).

**GPUs** (Claude's call, from boltz-2's history): every image stands on CUDA ≥ 12.8 wheels (a cu128 torch, not upstream's cu126) so the same image runs on Ampere, Ada, Hopper, and Blackwell (sm_100); boltz-2 1.0.1's cu126 base died on a B200 with "no kernel image is available". One record per tool, from the L40S; the skill's Compute says which GPU the record came from and that Blackwell is expected but untested.

**Kernel path** (Claude's call, from CONTRIBUTING's "The test"): the test runs the default kernel path on its 198-residue fixture, with cuEquivariance's fallback thresholds left alone, and the record comes from a GPU where the first-use compile fits (the L40S, as boltz-2 1.0.1's did). Raising the thresholds is not done in the test; if a Blackwell user hits boltz-2's compile cost, the Gotchas name the thresholds as the workaround.

## Approaches

For `opendde` and `openfold3` there was one sensible way: upstream ships a CLI, so the image makes it runnable and bakes in what it would otherwise download. For the ESMFold2 pair, which has only a Python API, three input formats were weighed for the shed's wrapper:

- **A. FASTA, proteins only,** chains joined by `:`. Least code; no ligands, nucleic acids, or MSAs.
- **B. JSON in upstream's own input schema** (proteins, DNA, RNA, ligands, modifications, MSAs, covalent bonds, pocket). **Chosen by Bryan.** Reading the `esm` 3.5.0 source showed this is cheaper than estimated: `esm.utils.structure.input_builder.deserialize_structure_prediction_input` already reads exactly this JSON into a `StructurePredictionInput`, so the wrapper adopts upstream's format rather than designing one.
- **C. Boltz-2's YAML,** for one input file across tools. Rejected: reimplements and tracks another project's format.

## The tools

### opendde

- **Upstream:** PyPI `opendde==1.2.0` with the `gpu` extra (torch 2.7.1, triton 3.3.1, cuEquivariance 0.10.0), Python 3.11–3.13. CLI `opendde pred`.
- **Base:** a CUDA 12.8 base with torch 2.7.1's cu128 wheels (upstream's Dockerfile uses 12.6; see Constraints, "GPUs"), carrying gcc for Triton's driver shim, as boltz-2 1.0.2 does. Devel or runtime image is the builder's call.
- **Weights and data:** `opendde.pt` from `aurekaresearch/OpenDDE` at revision `eddd563ce96571f784012edd8f045181c8f8627d`, plus the `common/` CCD files (`components.cif`, `components.cif.rdkit_mol.pkl`, `release_date_cache.json`, `obsolete_to_successor.json`), placed under `OPENDDE_ROOT_DIR` (set in the image, outside upstream's `VOLUME`). Prefer upstream's `scripts/download_opendde_data.sh --skip-search-database` at build time if it pins to the manifest; otherwise fetch the manifest's URLs and check its SHA-256s.
- **Commands the skill documents:** `opendde pred -i input.json -o out -n opendde_v1 --use_msa false --use_template false --use_rna_msa false`, plus seeds/samples/steps flags; the ColabFold MSA server (default, needs network) and A3M paths in the JSON as the MSA routes, untested.

### openfold3

- **Upstream:** PyPI `openfold3==0.5.1` with the `cuequivariance` extra; Python ≥ 3.12. CLI `run_openfold predict`. The default checkpoint in 0.5.1's source (`entry_points/parameters.py`) is `openbind-2025-06-30-174k`, file `of3-ob-2025-06-30-174k.pt` (~2 GB); the docs that name `openfold3_p2_v1` are stale.
- **Weights:** from the public AWS bucket (`s3://openfold3-data/openfold3-parameters/of3-ob-2025-06-30-174k.pt`, anonymous HTTPS), into `$OPENFOLD_CACHE` set in the image, so no Hugging Face login is needed. Its checksum is recorded in the Containerfile. The biotite CCD the package ships is used as is (no `setup_openfold` CCD update at run time).
- **Commands:** `run_openfold predict --query-json query.json --use-msa-server=False --output-dir out` with `--num-diffusion-samples`/`--num-model-seeds`; precomputed MSAs (`main_msa_file_paths`) and the ColabFold server documented, untested. Needs a ≥ 32 GB GPU per upstream.
- **Deferred unless the test needs it:** DeepSpeed's evoformer attention (needs CUTLASS); cuEquivariance only.

### esmfold2-fast and esmfold2

- **Upstream:** PyPI `esm==3.5.0` (torch ≥ 2.11, transformers ≥ 5.16, triton ≥ 3.6, Python ≥ 3.12); a PyTorch ≥ 2.11 CUDA runtime base (the shed's esmc-6b uses `pytorch:2.14.1-cuda13.0-cudnn9-runtime`).
- **Weights:** `biohub/ESMFold2-Fast` at `45fe8656f5b3ef493c17fcf9abe9a2968902e712` (25.1 GB over six fp32 safetensors; bundles its ESMC-6B backbone, no `ccd.pkl`) and `biohub/ESMFold2` at `69869f737beffec5294845ede23db5fc0b4f509e` (26.3 GB, bundles ESMC-6B, carries `ccd.pkl`). Both images take `ccd.pkl` from the `biohub/ESMFold2` revision above and point `ESMCFOLD_CCD_PATH` at it. **Precision (Bryan's decision):** the bundled ESMC-6B backbone's tensors are stored in bf16, the folding layers (trunk, structure, confidence, MSA encoder) in fp32 as published, roughly halving each image to ~15 GB. `esm` casts the backbone to fp8 at load anyway (fp8 GEMMs on compute capability ≥ 8.9, upcast elsewhere), so the stored bf16 adds one round-to-nearest-even step before that cast and nothing else. The model loads from a local directory (`resolve_model_dir` returns a directory with `config.json` as is); `config.json` is left unchanged, and the builder confirms that `esm` loads mixed-dtype shards without casting the trunk down.
- **Build of the weights:** esmc-6b's bounded-shard approach, copied into each tool: a committed shard plan cuts the pinned revision's six safetensors at tensor boundaries into spans of at most 256 MiB, each fetched, hashed, and written in its own `RUN`; backbone tensors are converted to bf16 inside that step (the fp32 bytes never reach a layer), others written as is; an index `RUN` writes `model.safetensors.index.json` for the new shards. The plan is computed once from the revision's safetensors headers (no 26 GB download needed to make it).
- **Wrapper `bin/esmfold2`** (identical in both tools, ~60 lines): `esmfold2 predict --in input.json --out out [--samples N] [--seed S] [--loops N] [--steps N]`. The input is a JSON object, or a list of them, each with a `name` and upstream's serialized `StructurePredictionInput` (`sequences` of `protein`/`dna`/`rna`/`ligand` entries, optional `covalent_bonds`, `pocket`); it calls `deserialize_structure_prediction_input`, then `ESMFold2InputBuilder().fold(...)`, and writes `out/<name>_sample_<i>.cif` and `out/<name>_sample_<i>.json` (`plddt` mean, `ptm`, `iptm`, `pair_chains_iptm`; per-residue pLDDT and PAE in `.npz`). The checkpoint loads once per invocation, so a batch is one call. MSAs: upstream's JSON form (`"msa": {"sequences": [...]}`); on `esmfold2-fast` the skill says MSAs are not used. Defaults for loops and steps follow the model card (3 loops, 50 steps for Fast; upstream's `fold()` defaults of 20 and 200 for the full model, cut if the 30-minute test budget needs it).

## Tests

Each test folds one complex on the far side of the first code-path threshold: HIV-1 protease homodimer (2 × 99 = 198 residues, boltz-2's `complex_large`), past cuEquivariance's 100-residue crossover in the two AF3-style models. Known answer, as boltz-2's `check.py`: two chains A and B, 198 CA atoms, finite confidences in range, mean pLDDT above a trained-weights floor (0.4 on a 0–1 scale, or 40 on 0–100). Offline, single-sequence, one sample. For `esmfold2`, a second input exercises the MSA path with a tiny hand-written alignment, since that is what separates it from Fast. The fallback thresholds are not raised (see Constraints, "Kernel path").

## Left to the builder

Exact base image per tool; exact package pins at build time; OpenDDE's output file names and confidence fields (read from its first run); the trained-weights floor per model (from the first real run, set with margin); whether the `fast` or `fused` kernel backend is the ESMFold2 default (whatever runs on an L40S without `opt_core`).
