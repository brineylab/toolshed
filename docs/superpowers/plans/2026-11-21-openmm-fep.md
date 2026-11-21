# openmm-fep 1.0.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `ghcr.io/brineylab/toolshed/openmm-fep:1.0.0`: binding ΔΔG of one antibody point mutation from an `openmm relax` complex, glycans and all, through a patched feflow 0.2.1, tested on a GPU with no network.

**Architecture:** One image: a pinned micromamba environment (OpenMM 8.4.0, openfe 1.13.0, feflow 0.2.1) on an NVIDIA CUDA 12.9 base, with `feflow-0.2.1.patch` applied, and one wrapper, `bin/openmm-fep`, with two verbs. `run` builds a leg's wild-type and mutant end states and atom mapping, then executes one whole feflow DAG (setup, K cycles, result) as one replicate. `analyze` pools every replicate's works per leg through feflow's `gather` and writes ΔG per leg and ΔΔG.

**Tech Stack:** OpenMM 8.4.0, openfe 1.13.0, gufe 1.13.1, feflow 0.2.1, Python 3.12.16, micromamba 2.9.0, podman, and flyloft direct jobs with `--tool name@sha256:<digest>`.

**Spec:** `docs/superpowers/specs/2026-11-21-openmm-fep-design.md` (approved 2026-11-21). Read it first; this plan argues from it. Also read `AGENTS.md`, `CONTRIBUTING.md`, `tools/openmm/` (the sibling: Containerfile layout, `bin/openmm`'s loader, `test/run.sh` + `test/check.py`), and `tools/igor/` (a tool carrying an upstream patch). The spike's scripts are in the session scratchpad, `fep-spike/spike.py` and `setup_only.py`. They are reference only, not to be copied blind.

## Global Constraints

- Every rule in `AGENTS.md` "Rules a tool must keep". The Containerfile ends with `COPY test /toolshed/test`.
- Base `docker.io/nvidia/cuda:12.9.1-base-ubuntu24.04`; micromamba 2.9.0's static binary by sha256, as `tools/openmm/Containerfile`.
- One conda-forge environment at `/opt/env`, every top-level package exact: `python=3.12.16 openmm=8.4.0 openfe=1.13.0 feflow=0.2.1 gufe=1.13.1 cuda-version=12.9`, plus the pytorch build the solve picks (the spike solved `pytorch 2.14.1 cuda129_mkl`), pinned exactly. Solve with `CONDA_OVERRIDE_CUDA=12.9`.
- `ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility HF_HUB_OFFLINE=1`. The test passes under `--network none`.
- Wrapper shebang `#!/usr/bin/env -S python3 -I`; program name `openmm-fep`; verbs `run` and `analyze` exactly as spec §5.
- Force field: `["amber14-all.xml", "amber14/tip3pfb.xml", "amber14/GLYCAM_06j-1.xml"]`, the list given to `forcefield_settings.forcefields` and to the `ForceField` used for `addHydrogens(pH=7.0)`.
- feflow settings: defaults except `forcefields`, `num_cycles = --cycles`, `alchemical_settings.explicit_charge_correction = True`, `engine_settings.compute_platform` = `"CUDA"` when OpenMM lists it, else `"CPU"` with one stderr line.
- `--cycles` default 25 until Task 3's measurement says otherwise.
- `execute_DAG` runs with shared and scratch under `$TMPDIR` (else `/tmp`), `keep_shared=True, keep_scratch=True`; the wrapper removes that directory itself after copying `result.json` out.
- Refusals exit 2 with one line on stderr naming the problem, before any directory is written.
- The patch has two hunks (spec §7). Nothing else of feflow, openfe or gufe is changed.
- The test fits one GPU and 30 minutes and writes only under `/tmp`.
- Commit messages are one sentence stating the behavior, ending with the session's Co-Authored-By line. Pushing a candidate, a flyloft job's first readback, and opening the PR all ask the human first. Later rebuild-and-push or resubmit cycles under an approved readback do not.

## Review Focus

1. **A replicate directory left by a different request** (another mutation, partner or input). Expected: refused, naming the field that differs; nothing deleted. Test in Task 2.
2. **Mutations at Gly** (no CB, HA2/HA3) **and to Pro** (no backbone H). Expected: the mapping maps only atoms both ends have; setup succeeds. Test in Task 2 (A:G38A, bound, `--cycles 1`); Pro is covered by the same rule and a unit check in `check.py` of the mapping for an L→P mutant built with no cycles.
3. **`--partner` that omits the mutated chain, or names a chain the file lacks.** Expected: refused. Test in Task 2.
4. **`analyze` with one leg missing, or with no finished replicate in a leg** (only crashed directories). Expected: refused, naming the leg. Test in Task 4.
5. **PDB input from `openmm relax`** (glycan links in CONECT). Expected: the same end states as from mmCIF. Test in Task 2: the fixture's PDB form, unbound, `--cycles 1`; check.py compares its wildtype/mutant atom counts and glycan residues to the mmCIF run's.

---

## Development loop (read before Task 1)

- Build locally: `podman build -t openmm-fep-dev tools/openmm-fep` (CPU only here; a cycle at default length takes over an hour on CPU, so only refusals and end-state checks run locally).
- Iterate without rebuilding by mounting the wrapper and test over the image's copies: `podman run --rm --network none -v "$PWD/tools/openmm-fep/bin/openmm-fep:/usr/local/bin/openmm-fep:ro" -v "$PWD/tools/openmm-fep/test:/toolshed/test:ro" openmm-fep-dev bash /toolshed/test/run.sh`.
- **GPU runs go through flyloft direct jobs on the candidate image.**
  - After Task 1, ask the human once, then commit, `scripts/build openmm-fep`, and `scripts/push openmm-fep`. The push prints `<registry>/openmm-fep@sha256:<digest>`.
  - Give the flyloft readback once (coreweave-barcelona, 1 B200, `--time 00:30:00`) and get a yes. Then submit from the worktree root:
    `flyloft job submit --cluster coreweave-barcelona --gpu-type b200 --gpus 1 --time 00:30:00 --tool openmm-fep@sha256:<digest> tools/openmm-fep/bin tools/openmm-fep/test -- "PATH=\$PWD/tools/openmm-fep/bin:\$PATH bash tools/openmm-fep/test/run.sh"`
  - Wait with `flyloft job wait <id>` in the background. Results are in `.flyloft/jobs/<slug>/`.
  - The host's proxy refuses bundles over ~1 MB (flyloft#333), so gzip anything large.
  - Rebuild and push only when the Containerfile or patch changes.
- gufe's own `rmtree` fails on the cluster's NFS. That is why the work directory goes on node-local temporary storage.

### Task 1: The image, with feflow patched

**Files:**
- Create: `tools/openmm-fep/Containerfile`, `tools/openmm-fep/feflow-0.2.1.patch`, `tools/openmm-fep/test/run.sh` (placeholder that exits 1)

**Interfaces:**
- Produces: an image where `python3` is `/opt/env`'s, `feflow`, `openfe`, `gufe`, `openmm` import, OpenMM is 8.4.0, and both patch hunks are in the installed files.

- [ ] **Step 1: Write the patch.**
  - It is a unified diff with paths `a/feflow/...` / `b/feflow/...`, applied with `patch -p1 -d /opt/env/lib/python3.12/site-packages`.
  - Hunk 1, `feflow/protocols/nonequilibrium_cycling.py`: in the `handle_alchemical_waters(...)` call, the last argument `solvent_comp_a` becomes `system_generator.forcefield`.
  - Hunk 2, `feflow/utils/hybrid_topology.py`: `_find_exception(force, index1, index2)` keeps its signature and return value (the parameter list, or `[]`). It looks the unordered pair up in a dict built once per force from `getExceptionParameters` over `range(force.getNumExceptions())`, cached on the factory or a module dict keyed by `id(force)` holding the force alive.
  - A header above the diff (patch ignores leading text) says:
    - what each hunk does and why;
    - the measured effect (HyHEL-10 build: >40 min to 4.4 min; fixture: 196 s to 14 s; system XML identical but the barostat seed);
    - the K44A failure it fixes;
    - that feflow `main` had neither fix on 2026-11-21.
- [ ] **Step 2: Write the Containerfile** in `tools/openmm`'s order, with the Global Constraints' base, micromamba, environment and `ENV`. Then:
  - `COPY feflow-0.2.1.patch`, and `patch -p1` (add the `patch` apt package, pinned like openmm's gcc) with a comment naming both fixes.
  - A build check, `python3 -I -c`: import the four packages; assert `openmm.__version__ == "8.4.0"`; assert `"system_generator.forcefield"` appears in the installed `handle_alchemical_waters` call; and run a vacuum `HybridTopologyFactory` build on feflow's own test data (`feflow.tests.data.capped_AAs`: ALA_capped → GLY_capped with `ala_to_gly_mapping.json`, if they ship in the conda package; otherwise two capped peptides built with Modeller) that finishes.
  - Copy the licenses (OpenMM's three, openfe, feflow, gufe) to `/opt/openmm-fep/licenses`, and the patch beside them.
  - `COPY test /toolshed/test` last.
- [ ] **Step 3: Build.** `podman build -t openmm-fep-dev tools/openmm-fep`. Expected: the check prints `8.4.0` and exits 0.
- [ ] **Step 4: Prove the check bites.** Build once with hunk 1 removed from the patch; expected: the build fails at the check. Restore it.
- [ ] **Step 5: Commit** — "openmm-fep's image pins OpenMM 8.4.0, openfe 1.13.0 and feflow 0.2.1, with feflow patched for its quadratic hybrid build and broken alchemical-water call." Then ask the human to approve the first `scripts/build openmm-fep && scripts/push openmm-fep` and the flyloft dev-loop readback.

### Task 2: `openmm-fep run`

**Files:**
- Create: `tools/openmm-fep/bin/openmm-fep`, `tools/openmm-fep/test/complex.cif`, `tools/openmm-fep/test/complex.pdb`, `tools/openmm-fep/test/check.py`
- Modify: `tools/openmm-fep/test/run.sh`, `tools/openmm-fep/Containerfile` (COPY the wrapper; `openmm-fep --help` as a build check)

**Interfaces:**
- Produces:
  - `OUT/<leg>/replicate-<N>/{run.json, result.json, wildtype.cif, mutant.cif, log.txt}`.
  - `run.json` keys: `mutation, leg, partner, cycles, replicate, input_sha256, versions, settings` (settings = feflow's settings as JSON).
  - `result.json` is `ProtocolDAGResult.to_dict()` serialized with gufe's `JSON_HANDLER.encoder`.
  - Exit codes: 0 done (or already done), 2 refused.

- [ ] **Step 1: Make the fixtures.** In the worktree root:
  - `podman run --rm --network none -v "$PWD/tools/openmm/test:/in:ro,Z" -v "$PWD/tools/openmm-fep/test:/out:Z" ghcr.io/brineylab/toolshed/openmm:1.0.0 openmm relax /in/complex.cif /out/r`, then move `r/complex.cif` up and remove `r`.
  - Make `complex.pdb` by `openmm relax` of a PDB conversion of the same input. relax writes the input's format, so convert openmm's fixture to PDB first with OpenMM's `PDBxFile`→`PDBFile.writeFile` inside the openmm image.
  - Write both commands in `run.sh`'s header.
- [ ] **Step 2: Write the test's `run` cases in `run.sh`** (`set -euo pipefail`, work under `mktemp -d /tmp/openmm-fep.XXXXXX`, `trap` cleanup). Each refusal is checked for exit 2 and its stderr word:
  1. A:L47A bound, replicate 0, `--cycles 1`.
  2. A:L47A unbound `--partner A`, replicate 0, `--cycles 1`.
  3. Case 1 again: exit 0, "already done" on stderr, `result.json` byte-identical (sha256 before and after).
  4. Delete case 2's `result.json`, rerun: redone, new `result.json`.
  5. A:K44A bound `--cycles 1` into its own OUT.
  6. A:G38A bound `--cycles 1` into its own OUT.
  7. `complex.pdb`, A:L47A unbound `--partner A` `--cycles 1` into its own OUT.
  8. Refusals:
     - `A:F47A` (wrong wild type);
     - `A:N17A` (glycosylated Asn);
     - `A:L47aA` (no such residue);
     - `A:L47L` (to itself);
     - `--partner B` with an A mutation;
     - `--partner A` with `--leg bound`;
     - unbound without `--partner`;
     - case 1's OUT with `--mutation A:L47G` (Review Focus 1: refused, "mutation" named, case 1's files unchanged).
- [ ] **Step 3: Write `check.py`'s `run` checks.** Each one is a function and asserts:
  - L47A `mutant.cif` vs `wildtype.cif`: residue A47 is ALA in mutant with heavy atoms {N, CA, C, O, CB} and H names {H, HA, HB1, HB2, HB3}; every other residue identical in atom names; atom count difference = 9 (Leu 19 atoms → Ala 10).
  - The six GLYCAM residues (`0YB`, `4YB`×2, `VMB`, `0MA`×2) are present in bound end states. In unbound `--partner A`, exactly chain A's glycan (the one on A:N17) is present, and chain B is absent.
  - Each `result.json` loads with gufe and has finite forward and reverse works.
  - K44A's `log.txt` names the alchemical water (Cl⁻).
  - The PDB run's end states match the mmCIF unbound run's residue names and atom counts.
  - Without cycles, a function in check.py builds the L47P mapping by importing the wrapper as a module (`importlib.machinery.SourceFileLoader("openmm_fep", "/usr/local/bin/openmm-fep")`). It asserts that `H` is unmapped and N, CA, C, O, CB, HA are mapped.
- [ ] **Step 4: Run locally to see it fail** (no wrapper yet). Expected: `openmm-fep: command not found`.
- [ ] **Step 5: Write `bin/openmm-fep` `run`.** Functions, in the wrapper, with these signatures:
  - `parse_mutation(text: str) -> tuple[str, str, str, str, str]` returns (chain, wt, resnum, icode, new), from `C:XnY` with an optional lowercase insertion code. Unknown letters are refused.
  - `load(path: Path) -> tuple[Topology, positions]`: mmCIF through `tools/openmm/bin/openmm`'s `_pdbx_by_auth` (copied with its docstring), PDB through `PDBFile`. It drops HOH and monatomic ions, saying how many on stderr.
  - `leg_structure(topology, positions, leg, partner) -> tuple[Topology, positions]`: for unbound, it keeps the partner chains plus glycan chains bonded (`topology.bonds()`) to them, transitively through sugar–sugar bonds.
  - `check_target(topology, chain, wt, resnum, icode)`: the refusals of spec §5 (missing chain or residue, wrong wild type, glycan-bearing N/S/T named `NLN`/`OLS`/`OLT` or bonded to a sugar, Cys with an SG–SG bond, first or last residue of its chain).
  - `mutate(topology, positions, target, new3) -> tuple[Topology, positions]`: spec §6's Modeller procedure, with `ForceField` from the Global Constraints.
  - `atom_mapping(wt_rdkit, mut_rdkit, target_key) -> dict[int, int]`: by (chain, resnum+icode, name). In the target, only N, CA, C, O, and H/HA/CB when both sides have them.
  - `main()` runs only under `if __name__ == "__main__":`, so check.py can import the wrapper.
  - `run(args)`:
    - Compare any existing `run.json` against the request and refuse on a difference.
    - Done when `result.json` exists. Otherwise empty the directory and start over, saying so.
    - Write the end states and `run.json`, build the `ChemicalSystem`s with `SolventComponent()`, apply the Global Constraints' settings, then `create`.
    - Run `execute_DAG(..., raise_error=True, keep_shared=True, keep_scratch=True)` under a `tempfile.mkdtemp(dir=$TMPDIR or /tmp)`.
    - Write `result.json` atomically (temp file, then rename), then remove the temp dir.
    - Log feflow's INFO records to `log.txt` and stderr.
- [ ] **Step 6: Run the refusals and the L47P mapping check locally** (CPU, mounted wrapper). Expected: those pass; cycle cases are not run locally.
- [ ] **Step 7: Run the whole test on a B200** through the flyloft dev loop. Expected: exit 0. Note the wall times for each case from the log.
- [ ] **Step 8: Commit** — "openmm-fep run builds a leg's wild-type and mutant end states from an openmm relax complex, glycans kept, and runs one feflow replicate that a rerun skips or redoes."

### Task 3: Measure HyHEL-10 and settle `--cycles`

**Files:**
- Modify: `tools/openmm-fep/bin/openmm-fep` (only if the default changes), the spec's §8 (one line with the numbers)

- [ ] **Step 1: Make the input.** Relax the HyHEL-10 Fab + HEL Boltz-2 model (the session scratchpad's `ship/fly/AB.cif`; not committed) with `openmm:1.0.0`.
- [ ] **Step 2: One flyloft job** on the candidate image (gzip the CIF): H:Y33A bound and unbound (`--partner H,L`), `--cycles 2` each, plus H:K… (a Lys in the H chain's CDRs, the builder picks) bound `--cycles 1` to time the charge-changing setup at size. Record setup time, per-cycle time and atoms per leg from `log.txt`.
- [ ] **Step 3: Decide.**
  - If 25 cycles of the bound leg plus setup exceed ~12 h, lower `--cycles`' default to what fits in ~12 h, and say in the skill how many replicates make feflow's 100.
  - If the charge-changing setup is over 30 min at this size, stop and bring the profile to the human (a third patch hunk needs their say).
- [ ] **Step 4: Commit** — "openmm-fep's --cycles default fits a HyHEL-10-sized replicate in N hours on a B200." (or, with no change, the spec line alone).

### Task 4: `openmm-fep analyze`

**Files:**
- Modify: `tools/openmm-fep/bin/openmm-fep`, `tools/openmm-fep/test/run.sh`, `tools/openmm-fep/test/check.py`

**Interfaces:**
- Consumes: Task 2's replicate layout and `run.json` keys.
- Produces: `TABLE`, tab-separated, header `leg	dg_kcal_mol	err_kcal_mol	replicates	cycles`, rows `bound`, `unbound`, `ddg` in that order. The `ddg` row's replicates/cycles columns are empty.

- [ ] **Step 1: Add the test cases.**
  - `analyze` over case 1+2's OUT: three rows, all finite; ddg = bound − unbound to 1e-9; the ddg error is `sqrt(e_b² + e_u²)`.
  - Over K44A's OUT (bound only): refused, naming `unbound`.
  - Over an OUT holding case 1's bound replicate and a copy of case 6's (G38A) bound replicate as `replicate-1`: refused, naming `mutation`.
  - Over an OUT whose only unbound replicate has no `result.json`: refused, naming `unbound`.
- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement `analyze(args)`.**
  - Read every `replicate-*/run.json` and `result.json`. Refuse on any disagreement in `input_sha256, mutation, partner, settings, versions` (cycles and replicate may differ).
  - Per leg, rebuild the protocol from the stored settings and call `gather` on the list of `ProtocolDAGResult.from_dict(...)`, then `get_estimate()` / `get_uncertainty()` in kcal/mol.
  - Count replicates, and cycles as the number of forward works.
  - Write the table atomically.
- [ ] **Step 4: Run locally on the Task 2 GPU job's OUT directories** (copy them back from `.flyloft/jobs/<slug>/out` by having `run.sh` copy its OUT trees to `$FLYLOFT_OUT` when set). Expected: pass. Then run the whole test on the B200 once.
- [ ] **Step 5: Commit** — "openmm-fep analyze pools each leg's replicates into one BAR estimate and writes ΔG per leg and ΔΔG with errors."

### Task 5: The skill, the record, the pull request

**Files:**
- Create: `tools/openmm-fep/SKILL.md`, `results/openmm-fep/1.0.0.txt`

- [ ] **Step 1: Write `SKILL.md`** in CONTRIBUTING's order, frontmatter `name: openmm-fep`, `metadata.version: 1.0.0`.
  - Commands: the three of spec §5 as full command lines, plus a Slurm array example of four replicates per leg.
  - What it computes: sign and meaning, the unbound leg's reference state, and the pooled BAR with bootstrap error.
  - feflow's settings with values, and the versions.
  - Gotchas:
    - the patch (both hunks, why, measured effect, upstream state);
    - feflow is experimental upstream;
    - charge changes go through an alchemical water;
    - the refusals;
    - waters and ions dropped;
    - CPU impractical;
    - one point mutation per run;
    - why it is not an `openmm` verb, and when it would fold back.
  - Compute: the test's B200 times and Task 3's HyHEL-10 numbers.
- [ ] **Step 2: Rules check.** `python3 -I scripts/check.py tools/openmm-fep`. Expected: no reasons.
- [ ] **Step 3: Commit; rebuild and push the final candidate.**
- [ ] **Step 4: The GPU test of the final candidate.**
  - Run the image's own test with no mounts: `--tool openmm-fep@sha256:<final digest>`, command `bash /toolshed/test/run.sh`, no bundle beyond a one-line script if flyloft requires a path.
  - Then write the record from the job's `slurm.out`: `scripts/record openmm-fep ghcr.io/brineylab/toolshed/openmm-fep@sha256:<digest> <exit> "flyloft job <slug> (coreweave-barcelona, 1 b200)" <node> <started> < .flyloft/jobs/<slug>/slurm.out`.
  - Commit the record — "openmm-fep 1.0.0's record: the test passed on a B200 through flyloft."
- [ ] **Step 4b: No-network proof.** flyloft jobs can reach the network, and the full test takes hours on CPU. So locally, under `podman run --network none` on the final image, run the refusal cases, the L47P mapping check, and case 2 (unbound, one cycle) on CPU. Expected: exit 0. Put its wall time and result in the PR body.
- [ ] **Step 5: Registry check.** `python3 -I scripts/check.py tools/openmm-fep --registry ghcr.io/brineylab/toolshed`. Expected: no reasons.
- [ ] **Step 6: Open the pull request** (human's OK first) with the spec and plan, `Closes #54`. Body line: "Built by Claude Code; reviewed by <human> before merge."
