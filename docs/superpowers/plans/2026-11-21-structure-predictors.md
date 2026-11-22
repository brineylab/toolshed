# Structure Predictors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four toolshed tools — `opendde`, `openfold3`, `esmfold2-fast`, `esmfold2` — each folding a multi-chain complex offline on one GPU, each published through its own pull request with a passing test record.

**Architecture:** Each tool is `tools/<name>/` (SKILL.md, Containerfile, test/), built on this machine with `scripts/build`, pushed with `scripts/push`, tested on the cluster's L40S with `scripts/test <name> srun ...`. `opendde` and `openfold3` run upstream's CLI; the ESMFold2 pair share one wrapper, `bin/esmfold2`, over upstream's own JSON input, and build their weights through esmc-6b's bounded-shard tooling with the `esmc.*` backbone converted to bf16.

**Tech Stack:** podman, CUDA ≥ 12.8 PyTorch images, PyPI `opendde==1.2.0`, `openfold3==0.5.1`, `esm==3.5.0`; Python stdlib + the image's own packages for test checks.

**Spec:** `docs/superpowers/specs/2026-11-21-structure-predictors-design.md` (read it first; AGENTS.md and CONTRIBUTING.md are binding).

## Global Constraints

- Every rule in AGENTS.md "Rules a tool must keep" applies; CONTRIBUTING.md's steps are the build/push/test/PR procedure.
- `metadata.version: 1.0.0` for each new tool; image `ghcr.io/brineylab/toolshed/<name>:1.0.0`.
- CUDA ≥ 12.8 wheels in every image (not upstream's cu126), gcc present for Triton's driver shim.
- `HF_HUB_OFFLINE=1` in every image (`TRANSFORMERS_OFFLINE=1` too in the ESMFold2 pair). Tests pass under `--network none`.
- Test fixture: HIV-1 protease homodimer, sequence `PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF` as chains A and B (198 residues), single-sequence, one sample, seed 0. cuEquivariance fallback thresholds are NOT raised.
- Known answer for every tool: chains exactly {A, B}; 198 CA atoms in the first sample's mmCIF; confidences finite and in range; mean pLDDT above a trained-weights floor set from the first real run with margin (boltz-2 used 0.4 on a 0–1 scale).
- Pins: OpenDDE weights `aurekaresearch/OpenDDE@eddd563ce96571f784012edd8f045181c8f8627d`; OpenFold3 `of3-ob-2025-06-30-174k.pt` (2,287,872,989 bytes) from `https://openfold3-data.s3.amazonaws.com/openfold3-parameters/`; `biohub/ESMFold2-Fast@45fe8656f5b3ef493c17fcf9abe9a2968902e712`; `biohub/ESMFold2@69869f737beffec5294845ede23db5fc0b4f509e` (also the source of `ccd.pkl` for both ESMFold2 tools).
- Each image copies upstream's license file(s) beside the weights; each SKILL.md states the weights' license.
- One branch and one PR per tool, in order; the design doc and this plan ride in the first (`opendde`). Commit messages are one sentence stating the behavior, ending with the `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` line. PR body says "Built by Claude Code; reviewed by Bryan Briney". Merge with `gh pr merge --merge`, by Bryan.
- Push (`scripts/push`) and opening a PR publish under Bryan's name: ask before the first of each per tool.
- This machine has no GPU. The cluster test is run by Bryan from a login node (`scripts/test <name> srun --partition=l40s --gres=gpu:1 --time=00:30:00`) or through flyloft as one job recorded with `scripts/record`; ask which at the first test.

## Review Focus

1. **A file with several jobs** (ESMFold2 wrapper): a list of inputs should fold each, loading the model once — pinned by the esmfold2-fast fixture holding two jobs (Task 5).
2. **A caller's environment under pyxis** (`HF_HOME`, `HF_HUB_CACHE`, `PYTHONPATH` set to the caller's): the tool must still load its baked weights — pinned by each `run.sh` exporting `HF_HOME=/tmp/nowhere HF_HUB_CACHE=/tmp/nowhere PYTHONPATH=/tmp/nowhere` before running (Tasks 1, 3, 5, 7).
3. **A malformed input** (ESMFold2 wrapper: a job without `name`, or an unknown entity `type`): exits non-zero with a one-line message naming the job, not a silent partial output — pinned by `run.sh` asserting `esmfold2 predict --in bad.json --out /tmp/bad` fails (Task 5).
4. **An output directory that already exists** (all four): a rerun into the same `--out`/`-o` must not die on existing files — pinned in the ESMFold2 wrapper by `mkdir(parents=True, exist_ok=True)` and a second invocation in `run.sh` (Task 5); for the upstream CLIs, documented in Gotchas from what the first run shows.
5. **Inputs larger than the fixture** (a ~450-residue antibody pair): memory and time grow past what the test measures — not pinned by a test (the budget is one GPU, 30 minutes); each skill's Compute states the fixture's measured time and peak memory and says larger inputs were not measured.

---

### Task 1: opendde — the tool directory and a built candidate

**Files:**
- Create: `tools/opendde/SKILL.md`, `tools/opendde/Containerfile`, `tools/opendde/LICENSE` (upstream Apache-2.0 text, verbatim), `tools/opendde/test/run.sh`, `tools/opendde/test/check.py`, `tools/opendde/test/hivpr.json`
- Already committed on branch `opendde`: the spec and this plan.

**Interfaces:**
- Produces: candidate `ghcr.io/brineylab/toolshed/opendde:src-<tree>`; `OPENDDE_ROOT_DIR=/opt/opendde` holding `checkpoint/opendde.pt` and `common/{components.cif,components.cif.rdkit_mol.pkl,release_date_cache.json,obsolete_to_successor.json}`.

- [ ] **Step 1: Write the fixture** `test/hivpr.json`:

```json
[{"name": "hivpr", "modelSeeds": [0],
  "sequences": [{"proteinChain": {"sequence": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF", "count": 2, "id": ["A", "B"]}}]}]
```

- [ ] **Step 2: Write `test/run.sh` and `test/check.py`**, after boltz-2's: `set -euo pipefail`, timestamps, `nvidia-smi -L`, the Review Focus 2 exports, then exactly the skill's command `opendde pred -i hivpr.json -o "$out" -n opendde_v1 --use_msa false --use_template false --use_rna_msa false --seeds 0 --sample 1`, then `python3 -I check.py "$out"`. `check.py` globs `"$out"/hivpr/seed_0/predictions/` for one mmCIF and its confidence JSON and asserts the Global Constraints' known answer (use `gemmi` if the image has it, else biotite, which opendde pins). The exact file names and confidence keys are read from the first run (Step 6) and fixed in `check.py` then.

- [ ] **Step 3: Write the Containerfile.** Environment first: a PyTorch 2.7.1 CUDA 12.8 base whose Python is 3.11–3.13 (`docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`, as boltz-2), `apt-get install gcc`, `pip install --no-cache-dir "opendde[gpu]==1.2.0"` with the cu128 torch kept (pin `torch==2.7.1` from the cu128 index so pip does not swap in cu126). `ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility OPENDDE_ROOT_DIR=/opt/opendde HF_HUB_OFFLINE=1`. Weights last: one `RUN` fetching the five files from `https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/` (`opendde.pt` and `common/*`) and asserting each sha256 from opendde 1.2.0's manifest and `opendde/config/dependency_url.py` (`opendde.pt` 7b826620…cc, `components.cif` bb31ae5c…a6, `components.cif.rdkit_mol.pkl` d1cfb71f…35, `obsolete_to_successor.json` 2bc08348…ad, `release_date_cache.json` 8b1ef12d…42 — take full hashes from those files). `COPY LICENSE /opt/opendde/`, then `COPY test /toolshed/test` last.

- [ ] **Step 4: Write SKILL.md** in CONTRIBUTING's section order. Commands: the test's `opendde pred` line; MSA routes (ColabFold server default when `--use_msa true`, needs network; `pairedMsaPath`/`unpairedMsaPath` A3Ms in the JSON) documented as untested. Gotchas: preview status, single-sequence accuracy cost, the 100-residue cuEquivariance crossover the fixture crosses, Apache-2.0 weights, `opendde_abag.pt` not carried. Compute numbers are placeholders until Task 2.

- [ ] **Step 5: Check the rules, commit, build.**

Run: `python3 -I scripts/check.py tools/opendde` → no rule errors (registry checks come later).
Commit: `git add tools/opendde && git commit` ("opendde 1.0.0 folds complexes offline with OpenDDE's general-purpose checkpoint").
Run: `scripts/build opendde` → prints the candidate name.

- [ ] **Step 6: Offline smoke on this machine (no GPU).**

Run: `podman run --rm --network none <candidate> bash -c 'opendde --help && ls -l /opt/opendde/checkpoint /opt/opendde/common && opendde pred -i /toolshed/test/hivpr.json -o /tmp/o -n opendde_v1 --use_msa false --use_template false --use_rna_msa false --seeds 0 --sample 1 --device cpu --step 2 --cycle 1; find /tmp/o'`
Expected: help prints; five files present; the CPU prediction downloads nothing (no network errors) and writes the output tree. Fix `check.py`'s file names and keys from that tree; run `python3 -I /toolshed/test/check.py /tmp/o` inside the same container with the pLDDT floor temporarily disabled to confirm parsing. If a CPU run is impractical, stop at `opendde --help` and fix names from the GPU run in Task 2. Commit fixes and rebuild.

### Task 2: opendde — push, cluster test, record, PR

**Interfaces:**
- Consumes: Task 1's committed `tools/opendde/` and built candidate.
- Produces: `results/opendde/1.0.0.txt`, PR "opendde 1.0.0".

- [ ] **Step 1: Ask Bryan to approve the push; then** `scripts/push opendde`. Expected: exit 0, candidate anonymously pullable.
- [ ] **Step 2: Cluster test** (Bryan from a login node, or flyloft — ask): `scripts/test opendde srun --partition=l40s --gres=gpu:1 --time=00:30:00`. Expected: record with `exit: 0`, an L40S line, and `check.py`'s `ok:` line.
- [ ] **Step 3: Set the floor and the numbers.** From the record: set `check.py`'s pLDDT floor with margin below the observed value, and the skill's Compute (load time, fold time, peak memory if printed, GPU). This changes `tools/opendde/`, so commit, rebuild, push, retest (Step 2). Expected: second record passes.
- [ ] **Step 4: Verify and open the PR.** Run `python3 -I scripts/check.py tools/opendde --registry ghcr.io/brineylab/toolshed` → passes. Commit the record, push the branch, ask Bryan, then `gh pr create` (body: what the tool does, the record's GPU and result, "Built by Claude Code; reviewed by Bryan Briney"). Wait for `check` to pass.

### Task 3: openfold3 — the tool directory and a built candidate

**Files:** (branch `openfold3` from `main` after opendde merges, or from `main` now if Bryan prefers parallel PRs)
- Create: `tools/openfold3/{SKILL.md,Containerfile,LICENSE}`, `tools/openfold3/test/{run.sh,check.py,hivpr.json}`

**Interfaces:**
- Produces: candidate `openfold3:src-<tree>`; `OPENFOLD_CACHE=/opt/openfold3` holding `of3-ob-2025-06-30-174k.pt`, found by `run_openfold predict` with no `--inference-ckpt-path`.

- [ ] **Step 1: Fixture** `test/hivpr.json`:

```json
{"queries": {"hivpr": {"use_msas": false, "chains": [{"molecule_type": "protein", "chain_ids": ["A", "B"], "sequence": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"}]}}}
```

- [ ] **Step 2: `run.sh` and `check.py`** as Task 1, running `run_openfold predict --query-json hivpr.json --use-msa-server=False --num-diffusion-samples 1 --num-model-seeds 1 --output-dir "$out"`. `check.py` reads `hivpr/seed_*/hivpr_seed_*_sample_1_model.cif` and `..._confidences_aggregated.json` (`avg_plddt`, `ptm`, `iptm`, `sample_ranking_score`, `has_clash`), names fixed from the first run.

- [ ] **Step 3: Containerfile.** Base: a PyTorch CUDA 12.8 runtime image whose Python is ≥ 3.12 (the 2.7.1 image ships 3.11 — choose the oldest pytorch/pytorch CUDA 12.8 tag with 3.12+, or install `openfold3` into a uv-managed 3.12 venv on the 2.7.1 base; builder's call, recorded in a Containerfile comment). gcc. `pip install "openfold3[cuequivariance]==0.5.1"` with every resolved package pinned (freeze once, pin the direct ones and cuEquivariance). `ENV NVIDIA_… OPENFOLD_CACHE=/opt/openfold3 HF_HUB_OFFLINE=1`. Weights last: one `RUN` fetching the checkpoint over HTTPS from the AWS URL, checking its size (2,287,872,989) and a sha256 computed on first fetch and then written into the Containerfile; write whatever `$OPENFOLD_CACHE/ckpt_root` the 0.5.1 lookup (`entry_points/validator.py::_try_default_ckpt_path`) needs so the default lookup finds it offline. `COPY LICENSE`, `COPY test` last.

- [ ] **Step 4: SKILL.md.** Commands: the test's line; `--num-diffusion-samples`, `--num-model-seeds`, `--runner-yaml` (`structure_format: pdb`, `low_mem` preset); MSA server (default **on** — every call without `--use-msa-server=False` needs the network) and precomputed MSAs (`main_msa_file_paths`) documented as untested. Compute: ≥ 32 GB GPU per upstream. Gotchas: the MSA-server default; checkpoint is OpenBind-0 (`openbind-2025-06-30-174k`), not the preview2 weights older docs name; DeepSpeed kernels not installed; Apache-2.0.

- [ ] **Step 5: Check, commit, build, offline smoke** as Task 1 Steps 5–6 (`run_openfold predict --help`; confirm no download attempt with `--network none`, CPU run if practical).

### Task 4: openfold3 — push, cluster test, record, PR

Same steps as Task 2 with `openfold3`.

### Task 5: esmfold2-fast — wrapper, weights tooling, built candidate

**Files:**
- Create: `tools/esmfold2-fast/bin/esmfold2`, `tools/esmfold2-fast/{SKILL.md,Containerfile,LICENSE.md}` (Biohub's, verbatim from the pinned revision of github.com/Biohub/esm), `tools/esmfold2-fast/fetch_files.py` (copy of esmc-6b's), `tools/esmfold2-fast/build_shards.py` (esmc-6b's, adapted), `tools/esmfold2-fast/make_shard_plan.py`, `tools/esmfold2-fast/shard_plan.json` (generated), `tools/esmfold2-fast/test/{run.sh,check.py,jobs.json,bad.json}`
- Modify: the spec's line about ESMFold2-Fast and MSAs only if Step 7 shows the wrapper must refuse them.

**Interfaces:**
- Produces (Task 7 copies all of this):
  - `esmfold2 predict --in <json> --out <dir> [--samples N=1] [--seed S=0] [--loops N] [--steps N]`; `--loops`/`--steps` default to `$ESMFOLD2_LOOPS`/`$ESMFOLD2_STEPS` set in the image (Fast: 3 / 50; full: 20 / 200).
  - Input: one JSON object or a list; each `{"name": str, "sequences": [...], "covalent_bonds"?: [...], "pocket"?: {...}}`, the rest exactly `esm.utils.structure.input_builder.deserialize_structure_prediction_input`'s schema.
  - Output per job and sample `i` (0-based): `<out>/<name>_sample_<i>.cif` (`result.complex.to_mmcif()`), `<name>_sample_<i>.json` (`{"plddt": mean, "ptm", "iptm", "pair_chains_iptm": [[...]]}`), `<name>_sample_<i>.npz` (`plddt` per residue, `pae`).
  - `make_shard_plan.py --repo R --revision V --convert-prefix esmc. --out shard_plan.json [--budget 268435456]`; `build_shards.py --plan P --shard K | --index` with `--model-dir` (default `$ESMFOLD2_MODEL`).
  - `ESMFOLD2_MODEL=/opt/model`, `ESMCFOLD_CCD_PATH=/opt/model/ccd.pkl`.

- [ ] **Step 1: Fixtures.** `test/jobs.json` is a list of two jobs (Review Focus 1): `hivpr` (`{"type": "protein", "id": ["A", "B"], "sequence": <HIV PR>}`) and `gcn4` (`{"type": "protein", "id": "A", "sequence": "RMKQLEDKVEELLSKNYHLENEVARLKKLVGER"}`). `test/bad.json`: `[{"sequences": [{"type": "protein", "id": "A", "sequence": "MKT"}]}]` (no `name`).

- [ ] **Step 2: `run.sh`**: Review Focus 2 exports; `esmfold2 predict --in jobs.json --out "$out" --seed 0`; a second identical call into the same `$out` (Review Focus 4); `if esmfold2 predict --in bad.json --out /tmp/bad; then exit 1; fi` (Review Focus 3); `python3 -I check.py "$out"`. `check.py` asserts for `hivpr_sample_0`: chains {A, B}, 198 CA, `ptm`/`iptm` in [0, 1], `pair_chains_iptm` 2×2, mean pLDDT above the floor; for `gcn4_sample_0`: chain {A}, 33 CA; `.npz` `plddt` length equals residue count.

- [ ] **Step 3: `bin/esmfold2`** (`#!/usr/bin/env -S python3 -I`, argparse, subcommand `predict`). Load once: `EsmFold2Model.from_pretrained(os.environ["ESMFOLD2_MODEL"], device="cuda").eval()`, kernel backend `"fused"` if it runs on the L40S without `opt_core`, else `None` (decide from the first run; comment why). Per job: validate `name` (exit 2 with `job <index>: missing "name"`), `deserialize_structure_prediction_input(job)`, `ESMFold2InputBuilder().fold(model, inp, num_loops=…, num_sampling_steps=…, num_diffusion_samples=…, seed=…)`, normalize a single result to a list, write the three files. `--out` created with `exist_ok=True`. Under `torch.inference_mode()`.

- [ ] **Step 4: `make_shard_plan.py`** produces a plan in esmc-6b's `shard_plan.json` schema (`tool, repo, revision, base_url, span_budget_bytes, files[], shards[], weight_map, total_size`) plus `"convert_prefix": "esmc."` and `"dtype_out": "bfloat16"`. It downloads each safetensors file of the revision (resumable, via `fetch_files.py`'s logic), checks the Hub LFS sha256 (`/api/models/<repo>?blobs=true` → `lfs.sha256`), cuts at tensor boundaries into spans ≤ budget, hashes each span. Run it on this machine into the scratchpad's download dir; commit only `shard_plan.json`. Expected: ~100 shards, `total_size` ≈ 25.1 GB.

- [ ] **Step 5: Adapt `build_shards.py`**: the conversion casts only tensors whose name starts with the plan's `convert_prefix` to bf16 and leaves the rest byte-identical; `verify_converted_shard` asserts every `esmc.*` tensor is `BF16` and every other keeps its plan dtype; `write_index` emits the new shards' weight map; drop `--config-dtype` (config.json is left as published). Self-check in the file's `__main__ --selftest`: a two-tensor fp32 safetensors (`esmc.x`, `trunk.y`) converted in a temp dir yields `BF16` and `F32`. Run `python3 -I tools/esmfold2-fast/build_shards.py --selftest` → passes.

- [ ] **Step 6: Containerfile.** Base `docker.io/pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime` (esmc-6b's; Python ≥ 3.12, torch ≥ 2.11) — confirm `python3 --version` ≥ 3.12. gcc. `pip install esm==3.5.0` with resolved direct dependencies pinned, torch kept from the base. `ENV NVIDIA_… HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ESMFOLD2_MODEL=/opt/model ESMCFOLD_CCD_PATH=/opt/model/ccd.pkl ESMFOLD2_LOOPS=3 ESMFOLD2_STEPS=50`. `COPY` the helpers and the plan, `COPY --chmod=755 bin/esmfold2 /usr/local/bin/`, `COPY LICENSE.md /opt/model/`. `fetch-files` of `config.json tokenizer.json tokenizer_config.json` from the Fast revision and of `ccd.pkl` from the ESMFold2 revision (`biohub/ESMFold2@69869f73…`). One `RUN build-shards --shard K` per shard (generate the lines from the plan; esmc-6b's Containerfile is the template), then `RUN build-shards --index`. `COPY test /toolshed/test` last.

- [ ] **Step 7: SKILL.md.** Commands: `esmfold2 predict --in jobs.json --out out --seed 0`; flags; the input schema with a protein, a ligand (`{"type": "ligand", "id": "L", "ccd": ["ATP"]}`), and a DNA example from upstream's dataclasses; outputs. Gotchas: single-sequence model (an `msa` in the input is not used — confirm from the model's config/`msa_encoder` absence and say what happens), bf16 backbone/fp32 trunk storage and the fp8 cast at load (fp8 GEMMs on compute capability ≥ 8.9), ligands/DNA/RNA documented but untested, MIT weights and Biohub's acceptable-use policy link, `ESMFOLD2_LOOPS`/`ESMFOLD2_STEPS`.

- [ ] **Step 8: Check, commit, build, offline smoke** as Task 1 Steps 5–6: `esmfold2 predict --help`; `ls /opt/model` shows the shards, index, config, tokenizer, ccd.pkl; a CPU load of the model is not attempted (25 GB) — the GPU run is the test.

### Task 6: esmfold2-fast — push, cluster test, record, PR

Same steps as Task 2 with `esmfold2-fast`. In Step 3, also fix the kernel-backend choice from the run, and record image size and pyxis import time in Compute.

### Task 7: esmfold2 — a copy with the MSA path

**Files:**
- Create: `tools/esmfold2/` as a copy of `tools/esmfold2-fast/` with: `shard_plan.json` regenerated for `biohub/ESMFold2@69869f737beffec5294845ede23db5fc0b4f509e`; Containerfile's revision URLs, `ESMFOLD2_LOOPS=20 ESMFOLD2_STEPS=200` (reduce only if the test budget forces it, and say so in Gotchas); skill's numbers and name; `test/jobs.json` gains a third job `hivpr_msa` whose protein entry carries `"msa": {"sequences": [<query>, <4 point-mutant variants of the query, same length, no gaps>]}` per chain.

**Interfaces:**
- Consumes: everything Task 5 produced, unchanged except the values listed.

- [ ] **Step 1: Copy, regenerate the plan** (`make_shard_plan.py --repo biohub/ESMFold2 --revision 69869f73… --convert-prefix esmc.`), edit the values above.
- [ ] **Step 2: `check.py`** adds `hivpr_msa_sample_0` with the same assertions as `hivpr`.
- [ ] **Step 3: SKILL.md**: the MSA form (`"msa": {"sequences": [...]}`, query first, aligned rows of equal length; whether A3M lowercase insertions are accepted, from `MSA.from_sequences`), and that MSAs are generated elsewhere.
- [ ] **Step 4: Check, commit, build, offline smoke** as Task 5 Step 8.

### Task 8: esmfold2 — push, cluster test, record, PR

Same steps as Task 2 with `esmfold2`.
