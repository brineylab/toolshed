---
name: openmm-fep
description: "Binding ΔΔG of one antibody point mutation, by feflow's nonequilibrium-cycling free-energy protocol on OpenMM 8.4: `openmm-fep run` (one replicate of the bound or the unbound leg of a complex prepared by `openmm relax`, N-glycans included) and `openmm-fep analyze` (pool the replicates into ΔG per leg and ΔΔG, in kcal/mol, with an error). Reach for it to ask whether a mutation in an antibody's paratope strengthens or weakens binding to its antigen. One mutation per run, not for ligands, and a GPU is required in practice."
metadata:
  version: 1.0.0
---
# openmm-fep

Image: `ghcr.io/brineylab/toolshed/openmm-fep:1.0.0`

OpenMM 8.4.0 with openfe 1.13.0, gufe 1.13.1 and feflow 0.2.1 (patched, see Gotchas), PyTorch 2.14.1 for CUDA 12.9 and Python 3.12.16: one conda-forge environment at `/opt/env`, first on `PATH`, on NVIDIA's `cuda:12.9.1-base-ubuntu24.04`. No weights. It is a separate tool from `openmm` because openfe 1.13 needs an OpenMM older than 8.5 and `openmm` 1.0.0 ships 8.6.1.

## Commands

```bash
openmm-fep run complex.cif fep/ --mutation H:Y33A --leg bound --replicate 0
openmm-fep run complex.cif fep/ --mutation H:Y33A --leg unbound --partner H,L --replicate 0
openmm-fep analyze fep/ ddg.tsv
```

**`openmm-fep run <complex> <out> --mutation C:XnY --leg bound|unbound [--partner CHAINS] --replicate N [--cycles K]`** runs one replicate of one leg on one GPU.
- `<complex>` is `openmm relax` output, `.cif` or `.pdb`, glycans kept (relax writes the input's format). Glycan links come from the file's bonds: mmCIF `_struct_conn`, or PDB `CONECT`.
- `C:XnY` is the chain, the wild-type one-letter code, the author residue number (with its insertion code in lower case if it has one), and the new one-letter code: `H:Y33A`, `H:S100aG`. The wild-type letter is checked against the structure.
- `bound` is the whole complex. `unbound` keeps only the chains named in `--partner` (comma-separated, and it must include the mutated chain) plus the glycans bonded to them. `--partner` is required for `unbound` and refused for `bound`: for an antibody mutation in the heavy chain of an H+L Fab bound to an antigen chain `A`, the unbound leg is `--partner H,L`.
- `--cycles` (default 25) is the number of nonequilibrium cycles this replicate runs. Replicates pool, so four replicates at 25 are one estimate from 100 cycles, feflow's default per estimate.
- `<out>/<leg>/replicate-<N>/` holds `run.json` (the request, the input's sha256, the package versions, the full feflow settings), `result.json` (feflow's `ProtocolDAGResult`), `wildtype.cif` and `mutant.cif` (the leg's end states before solvation), and `log.txt`.
- A replicate with a `result.json` is done: running it again says so and exits 0. A replicate directory without one is a crashed run: it is emptied and started again, saying so. So a requeued Slurm array task needs no flags. Nothing is checkpointed inside a replicate.
- A request that cannot run is refused with one line on stderr and exit 2, before anything is written (two refusals come later; see Gotchas).
- The platform is CUDA when one is present; otherwise CPU, with a line on stderr saying that is impractical beyond tests.
- gufe's working directories live under `$TMPDIR` (else `/tmp`) on the node and are deleted when the result is copied out: they hold tens of MB of hybrid system and trajectory per replicate and failed to clean up on NFS.

**`openmm-fep analyze <out> <table>`** pools every finished replicate of each leg and writes a TSV: columns `leg`, `dg_kcal_mol`, `err_kcal_mol`, `replicates`, `cycles`; rows `bound`, `unbound` and `ddg`. It needs a finished replicate of each leg and refuses a directory whose replicates disagree on input hash, mutation, settings or package versions (only `num_cycles` may differ), or on partner within a leg, naming the one that does. The `ddg` row leaves `replicates` and `cycles` empty. **`analyze` silently skips replicates with no `result.json`**: if array tasks crashed it pools the rest, and the `replicates` column is the only sign, so check it equals what you submitted.

Four replicates per leg as a Slurm array (eight tasks; tasks 0-3 are bound, 4-7 unbound), then the pool once the whole array has finished:

```bash
jid=$(sbatch --parsable --array=0-7 --gres=gpu:1 --time=24:00:00 --wrap='
  leg=bound; [ $SLURM_ARRAY_TASK_ID -ge 4 ] && leg=unbound
  rep=$((SLURM_ARRAY_TASK_ID % 4))
  [ $leg = bound ] && partner="" || partner="--partner H,L"
  srun --container-image=ghcr.io#brineylab/toolshed/openmm-fep:1.0.0 \
    --container-mounts="$PWD":/work --container-workdir=/work \
    openmm-fep run complex.cif fep/ --mutation H:Y33A --leg $leg $partner --replicate $rep')

sbatch --dependency=afterok:$jid --wrap='
  srun --container-image=ghcr.io#brineylab/toolshed/openmm-fep:1.0.0 \
    --container-mounts="$PWD":/work --container-workdir=/work \
    openmm-fep analyze fep/ ddg.tsv'
```

If a task fails, `afterok` holds the pool: resubmit the failed tasks (`--array=` with their numbers; finished replicates are skipped), then run `analyze`.

## What it computes

- **Sign and meaning.** Each leg's ΔG is wild type → mutant, in kcal/mol, as feflow's `get_estimate` returns it. ΔΔG = ΔG_bound − ΔG_unbound, so a positive ΔΔG means the mutation weakens binding. The `ddg` row's error is the two legs' errors added in quadrature.
- **The unbound leg** is the partner cut from the relaxed complex, not relaxed again by itself, as is usual for relative binding free energies. Conformational change on unbinding is not sampled beyond what the cycles reach.
- **Pooling.** feflow's `gather` concatenates the forward and reverse works of every replicate it is given. The estimate is one BAR (pymbar) over all the pooled cycles, not an average of per-replicate estimates, and the error is a bootstrap over them.
- **The mutant** keeps the wild type atom for atom. Only the target residue changes: its side chain is deleted beyond N, CA, C, O and CB (CB too when the new residue is Gly), the new side chain's heavy atoms are built by PDBFixer, and its hydrogens are placed by OpenMM's `addHydrogens` at pH 7. Every other atom is copied from the wild type, name and position. The wrapper's check of that copy can only fire on a duplicated chain, residue number and atom name in the input (which would also break the mapping, and is refused); the check that can genuinely fire is that the target's backbone N, CA, C, O have not moved. The mapping between the end states matches atoms by chain, residue and name; in the mutated residue it maps N, CA, C, O, and H, HA and CB where both end states have them (no CB for a mutation to Gly, no H for Pro). A new His gets the protonation `addHydrogens` picks; `run.json` does not record it, it is visible only in `mutant.cif`'s atom names. Every other residue's protonation state is the input's, preserved, not re-decided.
- **Charge-changing mutations** (K→A, D→N, ...) keep the box neutral by turning one water into a counter-ion (feflow's `explicit_charge_correction`, on here). Both legs change charge by the same amount, so both take that path. The wrapper writes the ion it expects to `log.txt`: `K44A` inserts Na⁺.
- **feflow's settings**, its defaults but for these: the force field is `amber14-all.xml`, `amber14/tip3pfb.xml`, `amber14/GLYCAM_06j-1.xml` (as `openmm`, so structures and MD are comparable), `num_cycles` is `--cycles`, `explicit_charge_correction` is on (feflow's default is off, which runs a charge change uncorrected with only a warning), and the platform is as above. The defaults it keeps: 300 K, 1 bar, 4 fs steps with hydrogen mass 3 amu, solvent padding 1.5 nm in a rhombic dodecahedron, 0.15 M NaCl, TIP3P-FB water, and per cycle 12,500 equilibrium steps then 12,500 nonequilibrium steps in each direction (50 ps each at 4 fs). `run.json` has them all.

## Inputs and outputs

- **In:** `openmm relax` output, `.cif` or `.pdb`, with GLYCAM-named glycans. Waters and monatomic ions in the input are dropped before the legs are built (counted on stderr), since feflow solvates and ionises itself.
- **Out:** a directory per replicate as above, and `analyze`'s TSV. `<out>` is one mutation of one input: a `run` into a replicate directory whose `run.json` disagrees with the request on mutation, partner, cycles, replicate or input is refused, its files untouched. Use one directory per mutation.

## Compute

One GPU per replicate; an array runs them in parallel. The image is CUDA 12.9: it needs an NVIDIA driver R575 or newer.

Measured on one NVIDIA B200 with the tool's own test, whose fixture is openmm's test complex after `openmm relax` (two protein chains of 53 and 28 residues and six glycan residues, 1,478 atoms before solvation). Each case is a whole `run` at `--cycles 1`, setup included:

| case | wall time |
| --- | --- |
| A:L47A, bound | 98 s |
| A:L47A, unbound (chain A) | 81 s |
| A:K44A, bound (charge-changing) | 96 s |
| A:G38A, bound | 90 s |
| A:L47A, unbound, PDB input | 74 s |
| the whole test: all runs, both `analyze` runs, 11 refusals | 835 s |

The fixture's hybrid system runs at about 335 ns/day, so one cycle at feflow's default length (four 12,500-step stages of 4 fs, 200 ps in all) takes about 52 s. The patched hybrid build took 14 s on the fixture.

HyHEL-10 Fab with hen lysozyme (8,409 atoms before solvation, solvated by feflow at 1.5 nm padding, in a rhombic dodecahedron, to 109,704 particles in the bound leg's hybrid system and 62,915 in the unbound leg's; the log reports this as `hybrid system: N particles`, once the run completes, so a run that dies during setup never logs it), `H:Y33A`: setup (the hybrid build, minimisation and equilibration) takes about 7.7 minutes bound and 4.5 minutes unbound on a B200, and one cycle takes about 3.2 minutes bound (180 s of dynamics, at 96 ns/day) and 1.9 minutes unbound, so a 25-cycle bound replicate takes about 1.5 hours and a 100-cycle estimate (four replicates at the default 25) about 5.8 hours of GPU time for the bound leg and 3.5 hours for the unbound, about 9 hours for both legs. A charge-changing mutation costs no more at this size (`H:K64A` bound: setup 7.6 minutes). The unpatched hybrid build of that system (solvated in the spike with openmm's 1.0 nm padded cube, ~140,000 atoms, larger than feflow's box) took over 40 minutes and never finished; the patched hybrid build alone took 4.4 minutes in the spike (5.3 minutes bound in the run above, within the 7.7-minute setup).

CPU is impractical: use it only to check that a request runs.

## Running it

On a laptop with a GPU, from the directory holding the complex:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/openmm-fep:1.0.0 \
  openmm-fep run complex.cif fep/ --mutation H:Y33A --leg bound --replicate 0 --cycles 5
```

On a Slurm cluster with pyxis, one replicate:

```bash
srun --gres=gpu:1 --time=24:00:00 --container-image=ghcr.io#brineylab/toolshed/openmm-fep:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work \
  openmm-fep run complex.cif fep/ --mutation H:Y33A --leg unbound --partner H,L --replicate 0
```

The array under Commands runs a whole estimate. Submit it again to redo any task that crashed or was cut off: finished replicates are skipped. `analyze` is cheap and runs on CPU.

## Gotchas

- **feflow is experimental upstream.** Treat a ΔΔG as an estimate to compare across mutations run the same way, and judge it by its error and cycle count, which `analyze` writes beside it. A ΔΔG without them is not a result.
- **One point mutation per run.** No double mutants, insertions or deletions, and no ligands or other small molecules.
- **It is not an `openmm` verb** because openfe 1.13 requires OpenMM older than 8.5 and `openmm` 1.0.0 ships 8.6.1 for its ML/MM. If openfe supports OpenMM 8.5 or later, this folds back into `openmm` as a `fep` verb and this tool stops at its last version.
- **Refusals** (exit 2, one line on stderr naming the problem), all before anything is written, except the last bullet's:
  - `run`: a mutation not of the form `CHAIN:XnY`; a letter that is not one of the 20 standard amino acids; a mutation to itself; `--partner` with `bound`, no `--partner` with `unbound`, a `--partner` that omits the mutated chain or names a chain that is not a protein chain of the structure; `--cycles` below 1; a complex that is not a file or not `.pdb`/`.cif`; a chain or residue not in the structure; a wrong wild-type letter; the first or last residue of a chain (its terminal atoms do not map); an Asn, Ser or Thr that carries a glycan (the glycan would be orphaned); a Cys in a disulfide; any residue with another covalent link off its side chain; a replicate directory whose `run.json` disagrees with the request.
  - `analyze`: replicates that disagree (above), and a leg with no finished replicate.
  - The exceptions: a mutant that differs from the wild type anywhere outside the target residue (an atom's name or position), and an atom mapping that is not one-to-one or leaves a wild-type atom outside the target unmapped (duplicated residue numbers or atom names in the input), are refused inside the build, after the replicate directory exists. They leave it behind, and a rerun treats that directory as a crashed run.
- **What the test covers and does not.** The fixture is small (1,478 atoms before solvation, a 53-residue and a 28-residue chain, six glycan residues on two chains, no waters or ions, no antibody numbering or CDRs), so it crosses none of the size thresholds a Fab-antigen complex does: that is only measured by hand (Compute). The test runs `--partner A` only, never a multi-chain partner such as `H,L`; no insertion-code mutation such as `S100aG` (only one that is absent); never the `--cycles` default; and it never drops a water or ion. It mutates to Ala, Gly and (end state only, no cycles) Pro, never to a residue larger than Ala. The refusals it does not exercise are a terminal residue, a disulfide Cys, a side-chain link, a bad one-letter code or mutation format, `--cycles 0`, a chain not in the structure, a non-protein partner chain, a file that is not `.pdb`/`.cif`, a missing file, a `run.json` disagreement on partner, cycles, replicate or input, the mutant-differs-outside-the-target refusal, and `analyze` refusing on partner, settings, versions or input; the refusals that are run are a wrong wild-type letter, a glycan, an absent residue, a mutation to itself, `--partner` misuse and a `run.json` disagreement on mutation, and `analyze` refusing on mutation and on a leg with no finished replicate. `analyze` skipping unfinished replicates is not run either. Those paths are read from the code, not run.
- **Growing a side chain is less reliable than truncating one.** The new side chain's heavy atoms are built by PDBFixer and only locally minimised, with everything else frozen: essentially one arbitrary rotamer, which 25 cycles will not relax if it is badly placed (Trp, Arg). Alanine and glycine scans are what the tool is tested on; a mutation to a larger residue runs but is untested.
- **`run.json` and feflow's version.** conda's feflow package reports itself as `0.1.0+unknown`; the image's feflow is 0.2.1 plus `feflow-0.2.1.patch`. `run.json`'s `versions` therefore also records `openmm-fep` (this tool's version), which is what identifies the patched image.
- **Protonation is the input's.** Hydrogens are not stripped and re-added, so the input's HID/HIE/HIP, CYX and the rest are what the legs use; the new residue alone gets `addHydrogens(pH=7)`.
- **Waters and ions in the input are dropped**, not kept: the legs are solvated afresh at 0.15 M NaCl. A structural water or metal ion that matters to the mutation is lost.
- **The error column varies a little between `analyze` runs** over the same data: feflow's `get_uncertainty()` bootstraps without a seed. The estimate does not.
- **One directory per mutation and input.** `analyze` requires every setting but `num_cycles` to match across replicates, so replicates of different inputs, mutations or partners cannot be pooled.
- **The patch.** feflow 0.2.1 has two bugs, both still unfixed on feflow `main` as of 2026-11-21, and the image applies a patch (`feflow-0.2.1.patch`, in the tool's directory and at `/opt/openmm-fep/licenses/`) that fixes them. Its Containerfile step checks that both hunks applied and that a small hybrid build finishes.
  - `HybridTopologyFactory._find_exception` scanned every nonbonded exception on every lookup, which is quadratic in system size. The patch indexes each force's exceptions once, as openfe's own copy does. The fixture's hybrid build went from 196 s to 14 s, and HyHEL-10's from over 40 minutes (it never finished) to 4.4 minutes. The hybrid system's XML is identical apart from the barostat's random seed, so the physics is unchanged.
  - feflow passes its `SolventComponent` where openfe 1.13's `handle_alchemical_waters` takes the force field, so every charge-changing mutation died in setup with `AttributeError: 'SolventComponent' object has no attribute 'createSystem'`. The patch passes `system_generator.forcefield`.
- **feflow's charge-correction log line names the wrong ion.** It is not patched. It indexes `{-1: positive_ion, 1: negative_ion}`, but gufe's charge difference is charge A minus charge B (+1 for K→A) and openfe picks a positive ion, so K→A really inserts Na⁺ while feflow's line says Cl⁻. The wrapper logs its own correct line (`alchemical water: ... one water becomes a Na+ ion`). Do not read the ion from feflow's raw log.
- **Replicates repeat setup.** Each `run` builds and equilibrates its leg itself, so the setup time under Compute is paid once per replicate. That costs little beside the cycles and keeps the tool on feflow's public interface.
- **CUDA is not deterministic**: the same replicate rerun on a GPU gives a statistically equal, not bitwise equal, result.
- The wrapper runs under `python3 -I`: a caller's `PYTHONPATH` or user site-packages never replace what the image tested. The environment sets `HF_HUB_OFFLINE=1`; the test passes with `--network none`.
- Licenses: OpenMM's, openfe's, feflow's and gufe's (MIT) are in the image at `/opt/openmm-fep/licenses/`, with the patch.
