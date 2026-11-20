# openmm: Design

Status: draft 2026-11-20, from a brainstorm with the human. Makes no amendment to `docs/DESIGN.md`: `openmm` is an ordinary tool (one image, a wrapper where upstream has no program, a test on its hardware), not a general-purpose image. Calls made in writing that nobody has discussed are marked **(call)**.

## 1. Why

The lab runs OpenMM today in one-off scripts, person by person, for several jobs: short relaxations of predicted structures, explicit-solvent MD of proteins (monomers, multimers, unbound antibodies, antibody–antigen complexes), binding ΔΔG for antibody point mutations, ML/MM on interface residues, and work on glycosylated antigens. The scripts differ in force field, preparation, and protocol, so their results are not comparable, and each one is rebuilt for every machine. A shed tool gives one tested environment, two standard protocols as verbs, and a documented library way for everything past them, the same on a laptop under podman and on a cluster under pyxis.

Structures come mostly from Boltz-2 and AlphaFold 3 (mmCIF or PDB, no hydrogens, PDB chemical-component names). When glycans are present they are usually on the antigen, not the antibody. Experimental glycosylated structures are rare and not a target.

## 2. Goals, non-goals, anti-goals

**Goals.**
- `tools/openmm/`, one image, following every rule in `AGENTS.md`.
- **1.0.0**: a wrapper `openmm` with two verbs, `relax` and `md`; glycan support in both; ML/MM (AIMNet2 on a chosen region inside a classical system) as a documented library-way script; the library way for everything else.
- **1.1.0**: a third verb, `fep`, for antibody point-mutation binding ΔΔG via feflow. Designed here (§6) so 1.0.0 leaves room for it; built and published after 1.0.0.
- `md`'s settings stay inside what NVIDIA's cuPotential plugin supports (§4), so a cuPotential sibling, if its license is ever cleared, is a swap of forces rather than a new protocol.

**Non-goals.**
- Small molecules and ligands (parameterization, OpenFF, GAFF, espaloma): deferred, board card #50.
- cuPotential. Its runtime is NVIDIA-proprietary: redistribution only inside "your application" (§1.1(d)), and §2.12 bars disclosing benchmark results without NVIDIA's written permission, which collides with every shed skill's published Compute measurements. Revisit only with NVIDIA's or counsel's answer.
- Enhanced sampling (openmm-plumed), stability ΔΔG, analysis of trajectories (RMSD, contacts, plots: "analysis helpers" are not in the shed), and anything that submits or sequences jobs.
- Experimental glycosylated structures, and glycans in `fep` (feflow cannot read them).

**Anti-goals: what would make this a failure even if it ran.**
- Glycans silently dropped, or silently left without hydrogens. PDBFixer does both by default (its `removeHeterogens()` keeps only polymer residues; its hydrogen step downloads unknown residues' definitions from RCSB and, offline, swallows the failure). Every glycan that goes in comes out parameterized, or the run stops naming the residue.
- Any run-time network access, including model weights fetched into `~/.cache` (pyxis passes the caller's home in, so a cache under `~` misses whatever the image baked).
- A `md` restart that duplicates or drops trajectory frames, or that only resumes on the same GPU model.
- A wrapper that grows into the science: protocol logic belongs to OpenMM, PDBFixer, openmm-ml, and feflow; the wrapper chooses defaults, prepares input, and writes files.

## 3. The image

**Environment: micromamba on conda-forge.** feflow (1.1.0) is distributed only on conda-forge, as are the newest pymbar and openmmforcefields; building 1.0.0 on the same footing means 1.1.0 adds packages rather than changing the install method. The base is an NVIDIA CUDA 12 runtime image on Ubuntu 24.04 (glibc, and it already sets `NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES`; the Containerfile sets both explicitly anyway), pinned like every shed base. A static `micromamba` binary, pinned by version and checksum, creates one environment from conda-forge with `cuda-version=12.x` and every top-level package at an exact version; the environment's `bin` is first on `PATH` so `python3` and `openmm` resolve to it **(call)**: exact top-level pins, as the pip tools do; a fully explicit lock file is the builder's option.

**Packages at 1.0.0** (latest on 2026-11-20; the builder re-checks at build time):

| Package | Version | Why |
|---|---|---|
| `openmm` | 8.6.1 | the engine; ships Amber14/19, CHARMM36 (with carbohydrates), and `amber14/GLYCAM_06j-1.xml` + `glycam-hydrogens.xml` |
| `pdbfixer` | 1.12 | missing heavy atoms in standard residues |
| `openmm-ml` | 1.8 | ML/MM via `createMixedSystem`; 1.8 runs models through `openmm.PythonForce` (OpenMM ≥ 8.6.1), so `openmm-torch` is not needed |
| `aimnet` | 0.2.6 | AIMNet2, the ML/MM model (§5) |
| `python` | 3.12 | |

openmmforcefields is **not** needed at 1.0.0: OpenMM 8.6.1 itself ships GLYCAM_06j and CHARMM36 carbohydrates. 1.1.0 adds `feflow` 0.2.x and whatever OpenFE stack it pins (`openfe`/`gufe` ≥ 1.10, `pymbar`).

**Weights.** AIMNet2's `.pt` files, fetched at build time from aimnet's own URLs and checked against the sha256s in aimnet's registry, stored at `/opt/openmm/models/aimnet2/`, and loaded by explicit path (`modelPath=`), never through `~/.cache/aimnet`. The image also sets `AIMNET_CACHE_DIR` to that directory as a second line of defence. Before building, the builder confirms the AIMNet2 weights' license against CONTRIBUTING's license rules (images are public); if it forbids redistribution, stop and bring it back, since the fallback model (AceFF or FeNNix-Bio1, both charge-aware) is a different choice of model, not the builder's.

**Offline.** Nothing in the image reaches the network at run time; the test passes with `--network none`. `HF_HUB_OFFLINE=1` is set (nothing uses the Hub at 1.0.0, but feflow's stack may import `huggingface_hub`).

**Python.** The wrapper starts with `#!/usr/bin/env -S python3 -I`, and every library-way example in the skill runs `python3 -I`.

**Containerfile order.** Base, micromamba, environment, a build-time import check (OpenMM's CUDA platform registers, ForceField loads `amber14-all.xml` + `amber14/tip3pfb.xml` + `amber14/GLYCAM_06j-1.xml`, openmm-ml imports AIMNet2's model class), weights, the upstream license files, the wrapper with a `--help` smoke check, then `COPY test /toolshed/test`.

## 4. The wrapper and its verbs (1.0.0)

`tools/openmm/bin/openmm`, one Python program, stdlib plus what the environment holds. Every verb picks the CUDA platform at mixed precision when a GPU is visible and CPU otherwise, prints a short timing breakdown to stderr, and takes `--seed N` (default 0) where randomness enters.

### Input preparation (shared by both verbs)

1. **Read** PDB or mmCIF (Boltz-2 and AlphaFold 3 outputs, as written). mmCIF `struct_conn` records supply covalent bonds where present.
2. **Protein**: PDBFixer adds missing heavy atoms to standard residues. No missing-residue modelling (predicted structures have none), no heterogen removal.
3. **Glycans**: the wrapper converts PDB chemical-component names to GLYCAM_06j's names, residue and atom:
   - Sugars: the N-glycan set that predictions carry: `NAG`, `BMA`, `MAN`, `FUC`, `GAL`, `SIA`, plus their other-anomer codes where GLYCAM has them (`NDG`, `FUL`, `GLA`, …) **(call)**: this set; any other carbohydrate residue stops the run with its name and chain, rather than being dropped.
   - Linkages: each sugar's GLYCAM name encodes which of its positions carry other sugars (`4YB`, `0MA`, `VMB`, …), so the wrapper reads the glycosidic bonds from `struct_conn`, or, when a file lacks them, infers them from distance (anomeric C1/C2 to an O or the Asn ND2 within a cutoff) **(call)**: distance fallback, cutoff the builder's.
   - Protein attachments: Asn → `NLN`, Ser → `OLS`, Thr → `OLT`.
   - Atom names: CCD → GLYCAM (e.g. the N-acetyl group's `C7/O7/C8` → `C2N/O2N/CME`).
   - The result carries explicit bonds for every inter-residue linkage, so ForceField template matching sees the connected graph.
   - **Outputs keep GLYCAM names** (the human, 2026-11-20): every structure the verbs write carries GLYCAM residue and atom names, glycosylated Asn/Ser/Thr included (`NLN`/`OLS`/`OLT`). Preparation therefore also accepts GLYCAM-named input unchanged, so `relax`'s output feeds `md`, and `md`'s outputs feed the ML/MM script, with no reverse mapping. The skill's Gotchas say that protein selections and sequence extraction in analysis code (mdtraj, MDAnalysis, antibody numbering) will skip `NLN`/`OLS`/`OLT`, and that code looking for `NAG`/`MAN` will find GLYCAM names instead.
4. **Hydrogens**: `Modeller.addHydrogens` at pH 7.0, after `Modeller.loadHydrogenDefinitions('glycam-hydrogens.xml')`. No PDBFixer hydrogen step, so nothing reaches for RCSB.
5. **Check**: every residue in the topology matched a template, and every glycan residue in the input is present in the output with hydrogens; otherwise stop with a message naming the residue.

Force field for both verbs **(call)**: `amber14-all.xml`, `amber14/tip3pfb.xml`, `amber14/GLYCAM_06j-1.xml`, the combination OpenMM's documentation gives for glycoproteins (GLYCAM_06j is parameterized alongside ff14SB). ff19SB with OPC is the alternative; it is a different protocol, so it would be a different version, not a flag.

### `openmm relax`

```bash
openmm relax in/ out/
```

`<in>` is one structure file or a directory of them (flat). For each: prepare (above), then minimize with harmonic restraints on heavy atoms to their input positions, in GBn2 implicit solvent (`implicit/gbn2.xml`) **(call)**: restrained, implicit-solvent minimization, the AlphaFold-relax idea on a modern force field; restraint strength and tolerance are the builder's, chosen so heavy atoms move a few tenths of an Å, not more. GBn2's parameters come from `GBSAGBn2Force.getStandardParameters(topology)`, and nobody has confirmed it handles GLYCAM sugar atoms; the builder checks this first. If it does not, glycosylated inputs relax with the same restraints in vacuum **(call)**, and the skill says which inputs take which path. Writes `out/<name>.pdb` (or `.cif`, the input's format) with hydrogens, and `out/relax.tsv`: `name`, `energy_before`, `energy_after` (kJ/mol, the unrestrained potential), `heavy_atom_rmsd` (Å). Structures in a directory are relaxed one after another in one process.

### `openmm md`

```bash
openmm md complex.cif out/ --ns 100
```

One structure in (any protein system: one chain, a multimer, an unbound antibody, a complex, with or without glycans), one directory out. `--ns` (production length, required) and `--seed` are the only options **(call)**: everything else is a fixed protocol, changed by version, not flag.

Protocol **(call, all of it)**:
- Prepare (above), then solvate in an orthorhombic box with 1.0 nm padding, neutralized, 0.15 M NaCl.
- PME, 1.0 nm cutoff, Ewald error tolerance 5×10⁻⁴; hydrogen mass repartitioning (1.5 amu), `HBonds` constraints, 4 fs step; `LangevinMiddleIntegrator` at 300 K, 1 ps⁻¹; `MonteCarloBarostat` at 1 bar.
- Minimize; 100 ps NVT then 1 ns NPT with heavy-atom restraints released over the NPT stage; then production, `--ns` nanoseconds, NPT.
- These settings are inside cuPotential's supported set (PME/LJPME only, Ewald tolerance 5×10⁻⁴, orthorhombic or reduced-triclinic box at least three cutoffs per side, `LangevinMiddleIntegrator`, `MonteCarloBarostat`), which is the goal in §2.

Writes to `out/`: `system.pdb` (the solvated starting topology), `equilibrated.pdb`, `trajectory.xtc` (protein and glycans only, every 100 ps **(call)**: solvent-stripped, which is what the lab's analyses use and what keeps files small), `log.csv` (step, time, potential energy, temperature, volume, speed), `final.pdb`, and `state.xml`.

**Resuming.** Run the same command again and it continues: if `out/state.xml` exists, the verb loads it (`Context.setState`, from the XML `State`, which is portable across GPU models and platforms, unlike `saveCheckpoint`'s binary), truncates `trajectory.xtc` and `log.csv` to the last step the state records, and runs to `--ns`. It saves `state.xml` atomically (write then rename) at every trajectory frame. A run already at `--ns` exits 0 doing nothing; a larger `--ns` extends it. Velocities and box come with the state, so the continuation is statistically the same run (not bitwise: CUDA reductions are not deterministic).

## 5. ML/MM (1.0.0, the library way)

No verb. The skill documents one script, run as `python3 -I mlmm.py md-out/ mlmm-out/ "A:30-35,B:98-104"`, which starts from an `md` output directory (its `system.pdb`, GLYCAM-named and solvated, and its `state.xml`), builds the classical system with `md`'s force field and settings, selects the ML region by chain and residue range, and calls `MLPotential('aimnet2', modelPath=...)` → `createMixedSystem(topology, system, atoms, linkAtomDistances=...)`, then runs a short NPT simulation. The skill states what openmm-ml 1.8 offers and what it does not:
- **Model: AIMNet2** (`aimnet2`, wB97M-D3, ensemble member 0 by default). It takes the region's total charge, which interface regions need (Asp, Glu, Lys, Arg), and covers H, C, N, O, S and the other protein elements. MACE-OFF ignores charge and is non-commercial; ANI-2x takes no charge. The script computes the region's formal charge from the topology and passes it.
- **Link atoms** (new in 1.8) let the region cut across backbone bonds.
- **Mechanical embedding only**: ML–MM electrostatics come from the classical force field; electrostatic embedding is upstream's future work. Said plainly in the skill's Gotchas.
- `interpolate=True` exposes `lambda_interpolate`, for anyone switching between the classical and ML/MM energy.

The ML region also accepts glycan residues, since the classical system is built the same way.

## 6. `fep` (1.1.0)

Binding ΔΔG for antibody point mutations, through feflow's `NonEquilibriumCyclingProtocol` with OpenFE `ProteinComponent`s (hybrid topology, charge-changing mutations via alchemical water). Two legs: the complex, and the antibody alone; ΔΔG is their difference.

```bash
openmm fep setup complex.cif H:Y33A fep/
openmm fep run fep/ complex --replicate 0
openmm fep run fep/ antibody --replicate 0
openmm fep analyze fep/ ddg.tsv
```

- `setup` prepares both legs (protein preparation as §4; the antibody leg is the complex's antibody chains, which `setup` takes as an argument or infers **(call, the builder settles which)**) and writes feflow's protocol inputs into `fep/`. Glycans are stripped with a warning on stderr: feflow cannot read glycosylated proteins.
- `run` runs one replicate (a set of nonequilibrium cycles) of one leg on one GPU. Replicates are independent: a Slurm array or flyloft runs them in parallel. The verb never submits jobs.
- `analyze` combines every replicate found into ΔG per leg and ΔΔG, with uncertainty, as a TSV.
- Defaults (cycle length, cycles per replicate) are feflow's, recorded in the skill.
- feflow is labelled experimental upstream; the skill says so. If its conda pins cannot coexist with 1.0.0's environment (openmm 8.6.1, openmm-ml 1.8, aimnet's PyTorch), the fallback is a separate tool, `openmm-fep`, decided then with the human.

## 7. The test

`test/run.sh` runs every documented command, under `/tmp`, with no network, on one GPU in under 30 minutes. Fixtures are kilobytes and cross the thresholds that matter:

- **A glycosylated fixture from Boltz-2** (the shed's own `boltz-2` tool, so its names and records are what users will feed in): a small glycoprotein or glycopeptide with a branched N-glycan (at least `NAG-NAG-BMA` with two `MAN` branches), so linkage-dependent names, branching, and the Asn attachment are all exercised. Known answers: every glycan residue present with hydrogens, the expected GLYCAM residue names, the expected atom count.
- **A distance-inferred variant**: the same structure with `struct_conn` removed, which must give the same GLYCAM names.
- **An unsupported-sugar fixture**: must exit nonzero naming the residue.
- `relax` on a directory of two structures: energies fall, heavy-atom RMSD under the bound.
- `md` on a small protein: `--ns` small enough for minutes; then rerun with a larger `--ns` to exercise resuming. Known answers: the trajectory's frame count and the log's last step equal what an uninterrupted run would give, with no duplicated times.
- The ML/MM script on a small two-chain system with a charged residue in the region, for a few hundred steps: finite energies, the ML force present, the passed charge matching the region's formal charge.
- 1.1.0 adds `fep` on a tiny system (a short peptide pair, one charge-preserving and one charge-changing mutation), few cycles: the pipeline runs and `analyze` writes finite numbers. It cannot check convergence; the skill says so.

## 8. The skill

Frontmatter `name: openmm`, `description`, `metadata.version: 1.0.0`. Body in CONTRIBUTING's order. Compute reports measured `md` throughput (ns/day) for the fixture and for one realistic system (an antibody–antigen complex at ~100k atoms) on the GPU the test ran on, and `relax` time per structure. Gotchas cover at least: the glycan sugar set and what happens outside it; GLYCAM names in every output (§4); mechanical embedding only; force field and water model fixed per version; resumption semantics; CUDA non-determinism; that cuPotential is absent and why (its license, not its speed); and that small molecules are not supported (card #50).

## 9. Left to the builder

Exact package versions at build time; the base image's exact tag; restraint strengths and tolerances in `relax`; the glycan distance cutoff; reporter intervals other than the trajectory's; how the wrapper is organized internally; fixture construction details within §7; the realistic system used for the Compute measurement.
