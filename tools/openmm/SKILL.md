---
name: openmm
description: "OpenMM for protein systems, N-glycans included, on Amber14, TIP3P-FB and GLYCAM_06j: `openmm relax` (restrained minimization in GBn2 implicit solvent, one structure or a directory) and `openmm md` (solvate, equilibrate, and run NPT production that a rerun resumes), plus ML/MM with AIMNet2 on an interface region and the OpenMM library itself under python3 -I. Reach for it to relax predicted structures from Boltz-2 or AlphaFold 3, to run explicit-solvent MD of monomers, multimers, antibodies and complexes, or to put a region under an ML potential. Not for small molecules or ligands."
metadata:
  version: 1.0.0
---
# openmm

Image: `ghcr.io/brineylab/toolshed/openmm:1.0.0`

OpenMM 8.6.1 with PDBFixer 1.12, openmm-ml 1.8, and aimnet 0.2.6 (with nvalchemi-toolkit-ops 0.5.0 and warp-lang 1.19.0), on PyTorch 2.14.1 for CUDA 12.9 and Python 3.12.16: one conda-forge environment at `/opt/env`, first on `PATH`, on NVIDIA's `cuda:12.9.1-base-ubuntu24.04`. The AIMNet2 weights (wB97M-D3, ensemble member 0) are in the image at `/opt/openmm/models/aimnet2/aimnet2_wb97m_d3_0.pt`, checked at build time against aimnet's registry sha256; nothing reaches the network at run time. Upstream ships no command-line program for these jobs, so the image carries the shed's wrapper, `openmm`, with two verbs and fixed protocols; ML/MM and everything past the verbs is the library way, `python3 -I script.py`, below. Both verbs and the ML/MM script run CUDA at mixed precision when a GPU is visible, else OpenMM's CPU platform.

## Commands

```bash
openmm relax in/ relaxed/
openmm md relaxed/complex.cif md/ --ns 100 --seed 1
```

**Preparation**, shared by both verbs, for every structure (PDB `.pdb` or mmCIF `.cif`, as Boltz-2 and AlphaFold 3 write them):
- PDBFixer adds missing heavy atoms to standard residues. No missing residues are modelled and no heterogens are removed.
- Hydrogens in the input are stripped, then OpenMM's `Modeller` adds them at pH 7.0.
- Glycans named by their PDB chemical components (`NAG`, `NDG`, `BMA`, `MAN`, `FUC`, `FUL`, `GAL`, `GLA`, `SIA`, `SLB`) are renamed to GLYCAM_06j, residue and atom (the N-acetyl's `C7/O7/C8` become `C2N/O2N/CME`). Their links come from the file's bonds (mmCIF `_struct_conn`, PDB `CONECT`) or, where it has none, from distance: an anomeric carbon with no link bonds to the nearest sugar hydroxyl O, Asn ND2, Ser OG, or Thr OG1 within 2.0 Å. Each sugar's GLYCAM name encodes its linked positions (`0YB`, `4YB`, `VMB`, `0MA`, ...), and the Asn, Ser, or Thr carrying one becomes `NLN`, `OLS`, or `OLT`. A linked sugar's leaving anomeric O is removed. stderr says how many links were found by distance.
- GLYCAM-named input is accepted unchanged, so `relax`'s output feeds `md`, and `md`'s feeds the ML/MM script.
- Any residue the force field has no template for stops the run, naming the file, the residue, its chain, and its number. Every glycan residue that goes in comes out with hydrogens, or the run stops.

The force field is fixed for the version: `amber14-all.xml`, `amber14/tip3pfb.xml`, `amber14/GLYCAM_06j-1.xml`.

**`openmm relax <in> <out>`.** `<in>` is one `.pdb`/`.cif` file or a flat directory of them; inputs that share a stem (`a.pdb`, `a.cif`) stop the run before anything is written. Each structure is prepared, then minimized in GBn2 implicit solvent (`implicit/gbn2.xml`, no cutoff, no constraints) with its heavy atoms held to their prepared positions by 10 kcal/mol/Å² (AlphaFold relax's stiffness), to OpenMM's default tolerance. Glycosylated structures take the same path. Structures in a directory run one after another in one process.

**`openmm md <structure> <out> --ns N [--seed S]`.** One structure in, one directory out. The protocol is fixed; `--ns` (production length, required) and `--seed` are the only options.
- Prepare, then solvate with OpenMM's `addSolvent(padding=1.0 nm)`: a cube whose side is the solute's bounding-sphere diameter plus 1 nm. At the start that leaves at least 1 nm between the solute and its own periodic image; it is not 1 nm from the solute to the box edge. TIP3P-FB water, neutralized, 0.15 M NaCl.
- PME with a 1.0 nm cutoff and Ewald tolerance 5×10⁻⁴; hydrogen mass repartitioning (1.5 amu) with `HBonds` constraints and a 4 fs step; `LangevinMiddleIntegrator` at 300 K, 1 ps⁻¹; `MonteCarloBarostat` at 1 bar every 25 steps.
- Equilibration: minimize with the solute's heavy atoms restrained at 10 kcal/mol/Å²; velocities at 300 K; 100 ps NVT restrained; 1 ns NPT in ten 100 ps segments whose restraint falls linearly to zero (the last runs unrestrained). Progress goes to stderr every 100 ps.
- Production: `--ns` nanoseconds of NPT, step and time counted from 0. Every 25,000 steps (100 ps) it writes a trajectory frame and a log row and saves `state.xml` atomically (write, then rename). `--ns` must be a positive multiple of 0.1.
- **Resuming.** Run the same command again. With `out/state.xml` present, `md` loads it, cuts `trajectory.dcd` and `log.csv` back to the step it records, and runs on to `--ns`: no frame duplicated or dropped, and an interruption loses at most the current 100 ps. A larger `--ns` extends a finished run; one at or below what is done exits 0, saying so on stderr, touching nothing. The XML state carries positions, velocities, and box, and resumes on any GPU model or on CPU. Equilibration is not resumable: a run stopped before production starts over on the rerun.
- **Seeds.** Without `--seed`, OpenMM picks the seeds. With `--seed S`, the initial velocities, the Langevin integrator, and the barostat take `S`; each resume reseeds the integrator and barostat with `S` plus the frame index it resumes at, so a restart does not replay the same noise.
- `out/input.sha256` holds the input's hash: a rerun with a different input into the same `out/` stops, touching nothing. So does an `out/` holding md outputs but no `input.sha256`.

The second way in is the library itself, under `python3 -I`. **ML/MM** is a script, not a verb: AIMNet2 on a chosen region of an `md` output, everything else in `md`'s force field, then a short NPT run. Regions are whole residues, `chain:first-last`, comma-separated, a single residue as `A:28-28`, named by the input's chain ids and residue numbers (mmCIF's `auth_asym_id` and `auth_seq_id`), the ids `topology.pdb` keeps:

```bash
python3 -I mlmm.py md/ mlmm/ "A:20-21,A:24-24,B:4-5"
```

`mlmm.py`:

```python
"""AIMNet2 on an interface region inside an `openmm md` system: python3 -I mlmm.py <md-out> <out> "A:20-21,B:4-5"."""
import csv
import sys
from pathlib import Path

import openmm
import warp
from openmm import unit
from openmm.app import PME, ForceField, HBonds, PDBFile, Simulation
from openmmml import MLPotential

N_STEPS = 1000  # 1 ps at 1 fs; raise it for a real run
MODEL = "/opt/openmm/models/aimnet2/aimnet2_wb97m_d3_0.pt"
warp.config.log_level = warp.LOG_WARNING  # no startup banner on stdout (warp has no env var for it)

md, out, region = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
pdb = PDBFile(str(md / "system.pdb"))
state = openmm.XmlSerializer.deserialize((md / "state.xml").read_text())

# system.pdb renames chains and renumbers residues; topology.pdb keeps the input's ids and is the
# solute only, which comes first in system.pdb, so a span is resolved there and shares atom indices
solute = PDBFile(str(md / "topology.pdb")).topology
if [(a.element, a.name) for a in solute.atoms()] != [(a.element, a.name) for a in pdb.topology.atoms()][:solute.getNumAtoms()]:
    sys.exit("topology.pdb's atoms are not the first atoms of system.pdb")
atoms = []
for item in region.split(","):  # whole residues from "chain:first-last", numbered as in the input
    chain, span = item.split(":")
    lo, hi = map(int, span.split("-"))
    hit = [r for r in solute.residues() if r.chain.id == chain and lo <= int(r.id) <= hi]
    if not hit:
        sys.exit(f"{item} matches no residue")
    if len({r.chain.index for r in hit}) > 1:
        sys.exit(f"{item} matches residues in more than one chain")
    atoms += [a.index for r in hit for a in r.atoms()]

# the force field and nonbonded settings of `openmm md`, without hydrogen mass repartitioning
ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml", "amber14/GLYCAM_06j-1.xml")
system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1.0 * unit.nanometer,
                         ewaldErrorTolerance=5e-4, constraints=HBonds)

# the region's net charge goes to AIMNet2, which needs an integer
nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
total = sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in atoms)
charge = round(total)
if abs(total - charge) > 0.1:
    sys.exit(f"region charge {total:.3f} is not near an integer: include each glycan whole, with its NLN/OLS/OLT")

# bonds cut at the region's edge are capped with link-atom virtual sites appended after the real
# atoms (openmm-ml's default distances); the returned topology includes them
mixed = MLPotential("aimnet", modelPath=MODEL).createMixedSystem(pdb.topology, system, atoms, charge=charge,
                                                                 returnInfo=True)
system, topology = mixed["system"], mixed["topology"]
natoms = pdb.topology.getNumAtoms()
system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))
integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond)

# CUDA at mixed precision when a GPU is present, else CPU
names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
platform, props = (openmm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"}) if "CUDA" in names \
    else (openmm.Platform.getPlatformByName("CPU"), {})
sim = Simulation(topology, system, integrator, platform, props)
sim.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
extra = [openmm.Vec3(0, 0, 0)] * (system.getNumParticles() - natoms)  # the link atoms
sim.context.setPositions(state.getPositions().value_in_unit(unit.nanometer) + extra)
sim.context.computeVirtualSites()
sim.context.setVelocities(state.getVelocities().value_in_unit(unit.nanometer / unit.picosecond) + extra)

with open(out / "log.csv", "w", newline="") as f:
    log = csv.writer(f)
    log.writerow(["step", "potential_kJ_mol", "temperature_K"])
    for _ in range(10):
        sim.step(N_STEPS // 10)
        s = sim.context.getState(getEnergy=True)
        kinetic = s.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        dof = 3 * natoms - system.getNumConstraints() - 3
        log.writerow([sim.currentStep, s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                      2 * kinetic / (dof * unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin))])
final = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
with open(out / "final.pdb", "w") as f:
    PDBFile.writeFile(pdb.topology, final.getPositions()[:natoms], f)
print(f"region_atoms {len(atoms)}")
print(f"region_charge {charge}")
```

- It rebuilds `md`'s system from `system.pdb` (without hydrogen mass repartitioning, at a 1 fs step), takes box, positions, and velocities from `state.xml`, and runs `N_STEPS` (1,000, 1 ps; raise it for a real run) at 300 K and 1 bar, logging every tenth. It prints `region_atoms` and `region_charge` on stdout.
- The region's net charge is the sum of its force-field charges, rounded and passed to AIMNet2; the script stops when the sum is not within 0.1 of an integer. It stops too, naming the span, when a span matches no residue or residues in more than one chain.
- AIMNet2 is loaded as `MLPotential("aimnet", modelPath=...)` from the image's file. Other model names (`"aimnet2"`, ...) go through aimnet's registry and its download cache, so never use them here.
- `createMixedSystem` uses openmm-ml's only embedding, mechanical, and caps bonds cut at the region's edge with link atoms at openmm-ml's default distances. `interpolate=True` adds a `lambda_interpolate` parameter for switching between the classical and the ML/MM energy.

**Continuing an `md` run your own way.** `system.pdb` and `state.xml` are all a script needs; this one runs on from `md`'s last state, writing a solvated trajectory frame and a stdout row (step, potential energy, temperature) every ps:

```bash
python3 -I continue.py md/ more.dcd 250000
```

`continue.py`:

```python
"""Continue an `openmm md` run in your own script: python3 -I continue.py <md-out> <out.dcd> <steps>."""
import sys
from pathlib import Path

import openmm
from openmm import unit
from openmm.app import PME, DCDReporter, ForceField, HBonds, PDBFile, Simulation, StateDataReporter

md, dcd, steps = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
pdb = PDBFile(str(md / "system.pdb"))
state = openmm.XmlSerializer.deserialize((md / "state.xml").read_text())

# md's force field and settings
ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml", "amber14/GLYCAM_06j-1.xml")
system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1.0 * unit.nanometer,
                         ewaldErrorTolerance=5e-4, constraints=HBonds, hydrogenMass=1.5 * unit.amu)
system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))
integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.004 * unit.picoseconds)
sim = Simulation(pdb.topology, system, integrator)

# box, positions and velocities from state.xml (not loadState: see Gotchas)
sim.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
sim.context.setPositions(state.getPositions())
sim.context.setVelocities(state.getVelocities())
sim.reporters.append(DCDReporter(dcd, 250))  # every ps, solvent included
sim.reporters.append(StateDataReporter(sys.stdout, 250, step=True, potentialEnergy=True, temperature=True))
sim.step(steps)
```

`Simulation` with no platform takes OpenMM's fastest, CUDA at its default single precision on a GPU; pass `openmm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"}` for what `md` runs. To prepare a structure the verbs' way inside a script, there is no import: run `openmm relax` first and load its output, which carries GLYCAM names and hydrogens.

## Inputs and outputs

- **In:** PDB (`.pdb`) or mmCIF (`.cif`); proteins of any number of chains, with N-glycans from the sugar set above. `relax` takes a file or a flat directory; `md` takes one file.
- **`relax` out:** `<out>/<stem>.pdb` or `.cif` (the input's format) per structure, with hydrogens, GLYCAM names, and the input's chain ids and residue numbers; and `<out>/relax.tsv`: `name` (the input's file name), `energy_before` and `energy_after` (kJ/mol, the unrestrained GBn2 potential of the prepared structure before and after), `heavy_atom_rmsd` (Å, unaligned, from the prepared positions).
- **`md` out:**
  - `system.pdb`: the solvated start, with OpenMM's own chain ids and residue numbers (a box holds more waters than PDB's residue field counts).
  - `topology.pdb`: the solute (everything but water, Na⁺, Cl⁻) with the input's chain ids and residue numbers; the atoms `trajectory.dcd` holds, and the first atoms of `system.pdb`, in the same order.
  - `trajectory.dcd`: the solute every 100 ps of production; frame k is step 25,000·k (no frame at 0). Load it with `topology.pdb`.
  - `log.csv`: production only, a row every 100 ps: step, time (ps), potential energy (kJ/mol), temperature (K), box volume (nm³), speed (ns/day).
  - `equilibrated.pdb` and `final.pdb`: the full system after equilibration and at the end of the last invocation, OpenMM's numbering.
  - `state.xml` (the resume point) and `input.sha256`.
- **ML/MM out:** `<out>/log.csv` (`step`, `potential_kJ_mol`, `temperature_K`; the potential includes AIMNet2's absolute energy of the region, about −2.4×10⁷ kJ/mol on the test's 400 atoms, so compare it only with itself), `<out>/final.pdb` (the system's real atoms in `system.pdb`'s numbering), and `region_atoms`, `region_charge` on stdout.

## Compute

One GPU. The image is 11.1 GB.

Measured on one NVIDIA B200 (driver 580) under pyxis, and on OpenMM's CPU platform on a 32-core laptop. Two systems: the test's ~23,600-atom box (the 1,478-atom glycosylated two-chain complex in a 6.29 nm cube), and a Boltz-2 prediction of HyHEL-10 Fab with hen lysozyme (PDB 3HFM's sequences, 558 residues), which solvates to 137,927 atoms in an 11.28 nm cube.

| job | system | B200 | CPU |
| --- | --- | --- | --- |
| `md` production | ~23,600 atoms | 1,280 ns/day | 36–40 ns/day |
| `md` production | 137,927 atoms | 472 ns/day | not measured |
| `md` equilibration (1.1 ns, every fresh run) | ~23,600 atoms | 78 s | 2,571 s |
| `md` equilibration | 137,927 atoms | 213 s | not measured |
| `relax`, per structure | the 1,478-atom complex | 4 s | ~3 s |
| `relax`, per structure | the 558-residue Fab–lysozyme | 13 s | not measured |
| ML/MM, 1,000 steps (1 ps) | a 400-atom AIMNet2 region in the ~23,600-atom box | 30 s, startup and Triton's first-use compile included | 1 min 46 s, startup included |

So 100 ns of a ~140k-atom antibody–antigen complex is about 5 hours on a B200; of the test's small complex, about 2 hours. The whole test took 62–68 minutes on CPU; its GPU time is in its record, `results/openmm/1.0.0.txt`.

## Running it

On a laptop, from the directory holding the structures:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/openmm:1.0.0 openmm relax in/ relaxed/
```

On a Slurm cluster with pyxis. When the job reaches its time limit, submit the same line again: it resumes from `md/state.xml`.

```bash
srun --gres=gpu:1 --time=24:00:00 --container-image=ghcr.io#brineylab/toolshed/openmm:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work \
  openmm md relaxed/complex.cif md/ --ns 100 --seed 1
```

The ML/MM script, saved as `mlmm.py` beside `md/`:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/openmm:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work \
  python3 -I mlmm.py md/ mlmm/ "A:20-21,A:24-24,B:4-5"
```

## Gotchas

- **The sugar set.** `NAG`, `NDG`, `BMA`, `MAN`, `FUC`, `FUL`, `GAL`, `GLA`, `SIA`, `SLB`, and their GLYCAM names. Any other carbohydrate stops the run, naming it, and so does a linkage GLYCAM_06j has no residue for (a multiply-linked sialic acid among them). GalNAc is not in the set, so most O-glycans stop.
- The test's glycans are `NAG`, `BMA`, and `MAN` on Asn. The other sugars' names and the Ser/Thr attachments come from GLYCAM's naming tables and the force field's templates; no fixture tests them.
- **GLYCAM names in every output.** Glycosylated Asn, Ser, and Thr are `NLN`, `OLS`, `OLT`, so protein selections and sequence extraction in analysis code (mdtraj, MDAnalysis, antibody numbering) skip them. Code looking for `NAG` or `MAN` finds `0YB`, `4YB`, `VMB`, `0MA` instead, with `C2N/O2N/CME` for the N-acetyl.
- Glycans relax in GBn2 like everything else: GBn2's parameters go by element, so sugar atoms get what protein atoms of the same element get. GBn2 was fitted on proteins and nucleic acids, not carbohydrates.
- Preparation strips the input's hydrogens and adds them again at pH 7, so protonation states are `Modeller`'s, never the input's.
- `relax` on a directory writes each structure and its `relax.tsv` row as it finishes, so a bad structure stops the run with the earlier ones already written.
- The force field and water model are fixed per version (Amber14 ff14SB, TIP3P-FB, GLYCAM_06j). ff19SB with OPC would be a new version, not a flag.
- The trajectory is wrapped into the box molecule by molecule, so a complex's chains can sit in different periodic images. Re-image before computing distances, contacts, or RMSD.
- `system.pdb`, `equilibrated.pdb`, and `final.pdb` use OpenMM's chain ids and residue numbers; only `topology.pdb` keeps the input's.
- `topology.pdb` is PDB, so a chain id longer than one character (mmCIF allows them, and a glycoprotein prediction with more than 26 chains has them) is written as a letter from the chain's index, and two chains can then share one; the ML/MM script stops on a span that then matches residues in both.
- Equilibration is not resumable: a run stopped before production starts over, deleting the partial outputs.
- `Simulation.loadState("md/state.xml")` fails (`invalid parameter name: k`): the state carries `md`'s global parameters (the restraint's `k`, the barostat's), so set box, positions, and velocities from it, as both scripts above do.
- **CUDA is not deterministic**: the same `--seed` on a GPU gives the same run statistically, not bitwise, and so does a resume.
- **ML/MM is mechanical embedding only**: ML–MM electrostatics come from the classical force field; openmm-ml 1.8 has no electrostatic embedding. Under periodic boundaries AIMNet2 switches its ML–ML Coulomb to DSF (aimnet warns so on stderr). Link atoms use openmm-ml's default distances.
- ML/MM spans use the input's chain ids and residue numbers (`topology.pdb`'s), not `system.pdb`'s. A span matching no residue, or residues in more than one chain, stops the run.
- GLYCAM's residue charges are not integers (±0.194 at linked positions), so a glycan goes into the ML region whole, with its `NLN`/`OLS`/`OLT`, or the script stops on the region's charge.
- The test's ML region is 400 atoms, past aimnet 0.2.6's 120-atom switch from a dense pair list to a neighbor list, so it exercises the path real interface regions take. Its complex has no `_struct_conn`, so its glycan links are found by distance; a copy with the six links in `_struct_conn` tests the bonded path, and another with the core's sugars in one chain, as AlphaFold 3 writes them, tests links named by `auth_seq_id`.
- The image is CUDA 12.9: it needs an NVIDIA driver R575 or newer.
- **No cuPotential.** NVIDIA's runtime may be redistributed only inside "your application" (its license §1.1(d)), and §2.12 bars publishing benchmark results without NVIDIA's written permission, which collides with every skill's Compute section. The reason is the license, not speed.
- **No small molecules or ligands** (parameterization, OpenFF, GAFF): deferred, board card #50. A ligand stops preparation, naming it.
- `fep`, antibody point-mutation binding ΔΔG through feflow, is planned as a separate tool, `openmm-fep` (brineylab/toolshed#54): feflow 0.2.1 needs an OpenMM older than 8.5, so it cannot install beside this image's 8.6.1.
- Licenses: OpenMM's are in the image at `/opt/openmm/licenses/` (MIT for the API and CPU platform, LGPL for the CUDA and OpenCL platforms, `Licenses.txt` for the rest), with PDBFixer's, openmm-ml's, and aimnet's (MIT each). AIMNet2's weights state no license of their own; their source, aimnetcentral, is MIT, so the image publishes them under the shed's rule for that case. AIMNet2 is Anstine, Zubatyuk & Isayev, "AIMNet2: a neural network potential to meet your neutral, charged, organic, and elemental-organic needs", *Chem. Sci.* 16, 10228–10244 (2025), doi:10.1039/D4SC08572H.
- The wrapper and both scripts run under `python3 -I`: a caller's `PYTHONPATH` or user site-packages never replace what the image tested.
