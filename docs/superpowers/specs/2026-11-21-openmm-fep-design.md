# openmm-fep 1.0.0 — design

Status: draft 2026-11-21, for approval. It replaces §6 of `2026-11-20-openmm-design.md` (`fep` as a verb of `openmm`), which became a separate tool (board #54).

## 1. Intent

The lab computes binding ΔΔG for antibody point mutations with one-off OpenMM scripts. `openmm-fep` makes that a shed tool. It takes a complex prepared by `openmm relax`, mutates one residue, and runs feflow's nonequilibrium-cycling free-energy protocol on two legs: the complex, and the mutated partner alone. ΔΔG_bind is the difference between the two legs.

It is a separate tool because feflow 0.2.1 needs openfe, and openfe 1.13.0 (the latest, checked 2026-11-21) needs an OpenMM older than 8.5. `openmm` 1.0.0 ships 8.6.1, which its ML/MM needs. When openfe supports OpenMM 8.5 or later, this becomes `openmm`'s `fep` verb and this tool stops at its last version.

## 2. What the spike showed (2026-11-21, board #54)

Scratch env: OpenMM 8.4.0, openfe 1.13.0, gufe 1.13.1, feflow 0.2.1, all from conda-forge.

- **Glycans need no special handling.** `openmm relax` output with GLYCAM-named glycans (openmm's test fixture: 6 sugar residues on A:N17 and B:N13) loads through `gufe.ProteinComponent.from_pdbx_file`, every link intact. feflow parameterises it with the force field below and runs cycles on CPU and on a B200. The 2026-11-20 design's plan to strip glycans is dropped.
- **feflow builds no end states.** The caller supplies the wild-type and mutant `ProteinComponent`s and a `LigandAtomMapping` between them. The spike built the mutant with OpenMM's Modeller (delete the side chain beyond CB, rename the residue, `addHydrogens`). The mapping matches atoms by chain, residue number and atom name. In the mutated residue it maps only the backbone and CB.
- **feflow 0.2.1's hybrid-topology build is quadratic.** `HybridTopologyFactory._find_exception` scans every nonbonded exception for each lookup; feflow `main` still does on 2026-11-21. Setup for HyHEL-10 Fab + HEL (~140k atoms solvated) was still in that loop after 40 min, on a B200 node and locally. Indexing each force's exceptions once, as openfe's own copy of the factory does, takes the build to 4.4 min. On the fixture it takes 196 s down to 14 s, and the hybrid system XML is identical except the barostat's random seed.
- **feflow's setup minimises the hybrid system** on the run's platform. On CPU that was ~66 min for HyHEL-10, so setup runs on the GPU like the cycles.
- **B200 throughput:** the fixture's hybrid system runs at ~335 ns/day, ~52 s per cycle at feflow's default cycle length (4 × 12,500 steps of 4 fs). HyHEL-10's per-cycle time was not reached and is measured during the build (§8).
- **feflow's DAG:** one `SetupUnit`, then `num_cycles` independent `CycleUnit`s, then a `ResultUnit`.
- **gufe's `execute_DAG` cleanup** (`shutil.rmtree` of its shared directory) failed on the cluster's NFS with "Directory not empty".
- **Charge-changing mutations are broken in feflow 0.2.1 against openfe 1.13.**
  - With `explicit_charge_correction` on, setup dies with `AttributeError: 'SolventComponent' object has no attribute 'createSystem'`. feflow passes its solvent component as the last argument of openfe's `handle_alchemical_waters`, where openfe 1.13 takes the force field. feflow `main` still does on 2026-11-21.
  - Passing `system_generator.forcefield` instead fixes it. A:K44A then turns one water into a Cl⁻ and runs a full cycle on CPU.
  - With the correction off, setup runs and only warns that a charge difference "is observed… please account for this in your final results".
  - The fixed K44A setup took 218 s against 14 s for L47A, unexplained; see §9.
- **`gather` pools.** `NonEquilibriumCyclingProtocol._gather` concatenates the forward and reverse works of every `ProtocolDAGResult` it is given. `get_estimate` runs BAR (pymbar) on the pooled works, `get_uncertainty` bootstraps, and both return kcal/mol. Replicates under approach 1 therefore pool their cycles into one estimate.

## 3. Goals, non-goals, anti-goals

**Goals**
- One point mutation's binding ΔΔG, with an uncertainty, from an `openmm relax` complex, glycans and all.
- Replicates that run independently and in parallel: a Slurm array, or flyloft jobs. A crash costs one replicate.
- The same force field as `openmm` (amber14-all, amber14/tip3pfb, amber14/GLYCAM_06j-1), so its structures and MD are comparable.
- feflow patched for the quadratic build and the broken alchemical-water call, the patch in the tool directory and documented.

**Non-goals**
- Several mutations in one transformation (double mutants), deferred until a question needs it.
- Small molecules (#50), absolute binding free energies, and solvation free energies.
- Preparing structures, which is `openmm relax`'s job.
- Submitting jobs. The tool runs one replicate; the caller schedules them.

**Anti-goals**
- A ΔΔG without its uncertainty and cycle count beside it.
- A silent change to feflow's physics. The patch changes speed only, and the build proves it is applied.
- A result built from replicates of different inputs, mutations or settings mixed together.

## 4. Approaches (chosen with the human, 2026-11-21)

1. **Each `run` is a whole feflow DAG (chosen).** One command builds both end states, runs setup and K cycles on one GPU, and writes that replicate's result. `analyze` pools every replicate of each leg. It uses only feflow's public interface (`create`, `execute_DAG`, `gather`). Setup is repeated per replicate: ~5 min of CPU hybrid build plus the GPU minimisation for an antibody–antigen complex, small against hours of cycles.
2. One setup per leg, then `run --cycles a-b` driving feflow's `CycleUnit`s by hand. This saves the repeated setup but couples the tool to gufe's unit internals, which change between releases. Not chosen.
3. Waiting for openfe to support OpenMM ≥8.5 and adding `fep` to `openmm`. Not chosen: no date upstream.

## 5. The interface

```bash
openmm-fep run complex.cif fep/ --mutation H:Y33A --leg bound --replicate 0
openmm-fep run complex.cif fep/ --mutation H:Y33A --leg unbound --partner H,L --replicate 0
openmm-fep analyze fep/ ddg.tsv
```

- **`COMPLEX`** is `openmm relax` output, `.cif` or `.pdb` (relax writes the input's format), read the way `openmm` reads it.
  - Glycan links come from the topology's bonds: mmCIF `_struct_conn`, or PDB `CONECT`, which OpenMM writes for non-standard residues.
  - Waters and monatomic ions in the input (relax removes no heterogens, so a crystal structure keeps them) are dropped before the legs are built, with a count on stderr, since feflow solvates and ionises itself **(call)**.
- **`run COMPLEX OUT --mutation C:XnY --leg bound|unbound [--partner CHAINS] --replicate N [--cycles K]`** runs one replicate of one leg.
  - `C:XnY` is chain, wild-type one-letter code, author residue number (with insertion code if any), new one-letter code: `H:Y33A`, `H:S100aG`. The wild-type letter is checked against the structure.
  - `bound` is the whole complex as given.
  - `unbound` keeps only the chains in `--partner`, which must include the mutated chain, plus the glycans linked to those chains. `--partner` is required for `unbound` and refused for `bound`.
  - `--cycles` defaults to 25 **(call: four replicates at feflow's default of 100 cycles per estimate, run in parallel)**.
  - The platform is CUDA when one is present, otherwise CPU, with a line on stderr saying CPU is impractical beyond tests.
- **Sign and meaning.** Each leg's ΔG is wild type → mutant, in kcal/mol, as feflow's `get_estimate` returns it. ΔΔG_bind = ΔG_bound − ΔG_unbound, so a positive value means the mutation weakens binding.
  - The unbound leg is the partner cut from the relaxed complex, unrelaxed on its own, as is usual for relative binding free energies. The skill says that conformational change on unbinding is not sampled beyond what the cycles reach.
  - The mutation changes the net charge by the same amount in both legs, so both legs take the same alchemical-water path.
- **`analyze OUT TABLE`** writes a TSV: one row per leg (`leg, dg_kcal_mol, err_kcal_mol, replicates, cycles`) and a `ddg` row (ΔG_bound − ΔG_unbound; error the two added in quadrature). It needs at least one finished replicate of each leg. It refuses a directory whose replicates disagree on input hash, mutation, partner or settings, naming the replicate that disagrees.
- **Output:** `OUT/<leg>/replicate-<N>/` holds:
  - `run.json`: the mutation, leg, partner, cycles, replicate, input sha256, versions, and the feflow settings as JSON.
  - `result.json`: feflow's `ProtocolDAGResult`.
  - `wildtype.cif` and `mutant.cif`, the leg's end states before solvation.
  - `log.txt`.
  - gufe's shared and scratch directories live under the node's local temporary directory (`$TMPDIR`, else `/tmp`) and are deleted after the result is copied out. That avoids the NFS failure and keeps hybrid pickles and trajectories (tens of MB per replicate) off shared storage **(call)**.
- **Restart:** a replicate with a `result.json` is done, and `run` says so and exits 0. A replicate directory without one is from a crashed run: `run` empties it and starts that replicate again, saying so. A Slurm requeue of an array task does the right thing with no flags **(call: feflow replicates are not checkpointed, so there is nothing finer to resume)**.
- Refused before any work, each with a one-line reason, exit 2:
  - a chain or residue not in the structure, or the wrong wild-type letter;
  - mutating a residue to itself;
  - an Asn, Ser or Thr that carries a glycan (the glycan would be orphaned);
  - a Cys in a disulfide;
  - a chain-terminal residue (its terminal atoms do not map) **(call)**;
  - a non-standard target letter;
  - an existing replicate directory whose `run.json` disagrees with the request.

## 6. Building the end states

- Read the complex with OpenMM's `PDBxFile`, using the same `auth_seq_id` re-keying `openmm` 1.0.0 uses for AF3-style glycan links. The builder copies that function, since tools do not import each other.
- For `unbound`, delete every chain not in `--partner`, and every glycan chain not linked (by struct_conn bond) to a kept chain.
- **Mutant:** Modeller deletes the target's atoms other than N, CA, C, O and CB (CB too when the new residue is Gly), renames the residue, and calls `addHydrogens` with the force field at pH 7. Every other residue already has its hydrogens. A new His gets the protonation `addHydrogens` chooses; it is visible only in `mutant.cif`'s atom names.
- **Mapping:** atoms are matched by (chain, residue, name). In the mutated residue only N, CA, C, O, and H, HA and CB where both ends have them, are mapped.
- **Settings:** feflow's defaults, with these overrides:
  - `forcefield_settings.forcefields` = the three files above;
  - `num_cycles` = `--cycles`;
  - `alchemical_settings.explicit_charge_correction = True`, so a charge-changing mutation (e.g. K→A) keeps the box neutral by turning a water into a counter-ion, which needs the patch's second hunk (§7). feflow's default is False, under which a charge change runs uncorrected, with only a warning **(call, per the 2026-11-20 design's "charge-changing mutations via alchemical water")**;
  - the platform as above.
  feflow's other defaults (300 K, 1 bar, 4 fs with hydrogen mass 3 amu, 1.5 nm rhombic-dodecahedral padding, 0.15 M NaCl, 12,500 equilibrium + 12,500 nonequilibrium steps per half-cycle) go in the skill with the versions.

## 7. The image

- Base `nvidia/cuda:12.9.1-base-ubuntu24.04` and micromamba 2.9.0, as `openmm`.
- One conda-forge environment with every top-level package exact: python 3.12.16, openmm 8.4.0, openfe 1.13.0, feflow 0.2.1, gufe 1.13.1, cuda-version 12.9. openfe pulls pytorch through openff-nagl, so pytorch is pinned to whatever build solves.
- No weights. `HF_HUB_OFFLINE=1` is set anyway, and the test passes with `--network none`.
- **The patch:** `tools/openmm-fep/feflow-0.2.1.patch`, a unified diff with two hunks. It follows igor's precedent (`tools/igor/igor-1.4.0.patch`).
  - In `feflow/protocols/nonequilibrium_cycling.py`, the `handle_alchemical_waters` call passes `system_generator.forcefield` where it passed `solvent_comp_a`, matching openfe 1.13's signature.
  - In `feflow/utils/hybrid_topology.py`, `_find_exception` looks pairs up in a per-force dictionary built on first use, keyed by the unordered atom pair, as openfe's `_generate_dict_from_exceptions` does. OpenMM forbids duplicate exception pairs, so the lookup returns what the scan returned. The searched forces are the old and new systems', which the factory never modifies.
  - The patch's header says what, why, the measured effect, and the upstream state.
  - The Containerfile applies it with `patch -p1` in site-packages and says why in a comment.
  - A build check proves both hunks applied: the patched lines are in the installed files, and a small hybrid build finishes.
  - The skill's Gotchas section describes it.
  - If feflow fixes it upstream, a later version drops the patch.
- Licenses: OpenMM's, and openfe's, feflow's and gufe's (MIT), beside the code.

## 8. The test

On one GPU, within 30 minutes, writing only under `/tmp`. The fixture is openmm's test complex after `openmm relax`, committed as `test/complex.cif` (~200 KB), with the command that made it.

- `run` A:L47A, `bound` and `unbound` (`--partner A`), one replicate each at `--cycles 1`. Check:
  - mutant atom counts (Leu's side chain gone, Ala's three HB present);
  - the glycans still present in both legs' end states;
  - finite work values.
- `analyze` gives three rows with finite values.
- `run` again on a finished replicate: no-op, exit 0.
- A crashed replicate (a directory with `run.json` and no result) is redone.
- A charge-changing mutation (A:K44A, bound, `--cycles 1`) runs, proving the alchemical-water path **(the builder picks a lysine away from the glycan)**.
- Refusals: wrong wild-type letter; A:N17 (glycosylated Asn); `--partner` with `bound`; `analyze` over replicates of two mutations.
- The test's GPU timings go in the record.
- Both mutations exist in the fixture: chain A has Leu47 and Lys44, 27–30 residues along the chain from its glycan on Asn17. Chain A alone (53 residues plus its glycans) is the unbound leg.

**Early, before the defaults are final:** the skill's Compute section also gives HyHEL-10 Fab + HEL (H:Y33A, both legs, a few cycles): per-cycle time and setup time on a B200. It is run through flyloft as a direct job on the candidate image (`--tool openmm-fep@sha256:…`), which flyloft now allows for unpublished tools. The GPU test can run the same way, with its record written by `scripts/record`. If a 25-cycle HyHEL-10 replicate would not fit well inside flyloft's 24-hour job limit, `--cycles`' default comes down to fit, and the skill says how many replicates make up feflow's 100.

**Measured (2026-11-22, B200, job 48d8bc21):** HyHEL-10 H:Y33A solvates to 109,704 particles bound (62,915 unbound); setup 7.7 min bound (4.5 unbound), 3.2 min per cycle bound (1.9 unbound), so a 25-cycle bound replicate is about 1.5 h and 100 cycles about 5.8 h (bound) + 3.5 h (unbound); the charge-changing H:K64A setup is 7.6 min, no slower than neutral. `--cycles` stays 25.

## 9. Left to the builder

- Whether `analyze` reports feflow's own uncertainty or a bootstrap over cycles, and how feflow's `gather` pools several DAG results. Check that results from two replicates pool their cycles rather than averaging their estimates, and say in the skill which it is.
- The exact stderr and log wording, and the `run.json` field names.
- How `unbound` finds glycans linked to kept chains.
- (Settled by §8's measurement: at HyHEL-10 size the charge-changing setup is 7.6 min, no slower than neutral, so no third hunk.) Why the charge-changing setup took 218 s against 14 s (another quadratic path, perhaps in the alchemical-water or mapping code), and whether it matters at antibody–antigen size. Measure it with the HyHEL-10 run. A third patch hunk is in scope only if it does matter, and comes back to the human first.
