#!/usr/bin/env bash
# openmm's test. Fixtures: complex.cif (Boltz-2, two chains, a branched
# N-glycan core on B:13 and a lone GlcNAc on A:17, no _struct_conn, so the
# glycan links are found by distance) and water-stripped 1UBQ. Runs:
# `openmm relax` on a directory of both; relax's GLYCAM-named output again,
# which must come back the same; the complex again with its six links
# written into a _struct_conn loop, which must give the same GLYCAM names;
# the same with the core's sugars in one chain, as AlphaFold 3 writes them,
# which must too; the complex with its first MAN renamed GLC, which must stop
# naming GLC; a directory where a.pdb and a.cif share a stem, which must stop
# before writing anything; and the input directory as the output, which must
# stop before writing anything. Then `openmm md` on relax's complex: 0.2 ns, extended to
# 0.4; rerun at 0.3, which must touch nothing; a crash simulated between a
# frame and its state.xml (one extra frame and log row), resumed to 0.5; and
# 1UBQ into the same directory, which must stop touching nothing. Then the
# skill's continue.py for 2 ps from md's state, and mlmm.py on the md output
# over check.py's ML_REGION. check.py verifies the answers. Writes only under /tmp.
set -euo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d /tmp/openmm.XXXXXX)
trap 'rm -rf "$T"' EXIT

mkdir -p "$T/in"
cp complex.cif 1UBQ.pdb "$T/in/"
# stderr is kept (and still shown) so check.py can count the links found by distance
openmm relax "$T/in" "$T/relax" 2>&1 | tee "$T/relax.err" >&2

# relax's own output, GLYCAM-named, goes back in unchanged (the input md takes)
openmm relax "$T/relax/complex.cif" "$T/rerelax"

# the bonded path: the same file with the six links in _struct_conn
mkdir -p "$T/withbonds"
python3 -I - complex.cif "$T/withbonds/complex.cif" <<'PY'
import sys

links = [  # (chain, comp, label_seq_id, atom) pairs; sugars have label_seq_id "."
    ("A", "ASN", "17", "ND2", "C", "NAG", ".", "C1"),
    ("B", "ASN", "13", "ND2", "D", "NAG", ".", "C1"),
    ("D", "NAG", ".", "O4", "E", "NAG", ".", "C1"),
    ("E", "NAG", ".", "O4", "F", "BMA", ".", "C1"),
    ("F", "BMA", ".", "O3", "G", "MAN", ".", "C1"),
    ("F", "BMA", ".", "O6", "H", "MAN", ".", "C1"),
]
cols = ["id", "conn_type_id"] + [f"ptnr{p}_label_{k}" for p in (1, 2)
                                 for k in ("asym_id", "comp_id", "seq_id", "atom_id")]
text = open(sys.argv[1]).read()
text += "loop_\n" + "".join(f"_struct_conn.{c}\n" for c in cols)
text += "".join(f"link{i} covale {' '.join(link)}\n" for i, link in enumerate(links, 1)) + "#\n"
open(sys.argv[2], "w").write(text)
PY
openmm relax "$T/withbonds/complex.cif" "$T/relax-withbonds" 2>&1 | tee "$T/withbonds.err" >&2

# AlphaFold 3's and wwPDB's layout: the core's five sugars in one chain, D, numbered 1-5 by
# auth_seq_id with label_seq_id ".", and _struct_conn naming each partner by both
mkdir -p "$T/af3"
python3 -I - complex.cif "$T/af3/complex.cif" <<'PY'
import sys

number = {"D": "1", "E": "2", "F": "3", "G": "4", "H": "5"}
out = []
for line in open(sys.argv[1]):
    f = line.split()
    if line.startswith("HETATM") and f[9] in number:  # columns 7 auth_seq_id, 9 label_asym_id, 15 auth_asym_id
        f[7], f[9], f[15] = number[f[9]], "D", "D"
        line = " ".join(f) + "\n"
    out.append(line)
links = [  # (chain, comp, label_seq_id, auth_seq_id, atom) pairs
    ("A", "ASN", "17", "17", "ND2", "C", "NAG", ".", "1", "C1"),
    ("B", "ASN", "13", "13", "ND2", "D", "NAG", ".", "1", "C1"),
    ("D", "NAG", ".", "1", "O4", "D", "NAG", ".", "2", "C1"),
    ("D", "NAG", ".", "2", "O4", "D", "BMA", ".", "3", "C1"),
    ("D", "BMA", ".", "3", "O3", "D", "MAN", ".", "4", "C1"),
    ("D", "BMA", ".", "3", "O6", "D", "MAN", ".", "5", "C1"),
]
cols = ["id", "conn_type_id"] + [f"ptnr{p}_{k}" for p in (1, 2) for k in
                                 ("label_asym_id", "label_comp_id", "label_seq_id", "auth_seq_id", "label_atom_id")]
text = "".join(out) + "loop_\n" + "".join(f"_struct_conn.{c}\n" for c in cols)
text += "".join(f"covale{i} covale {' '.join(link)}\n" for i, link in enumerate(links, 1)) + "#\n"
open(sys.argv[2], "w").write(text)
PY
openmm relax "$T/af3/complex.cif" "$T/relax-af3" 2>&1 | tee "$T/af3.err" >&2

# an unsupported sugar: the first MAN (chain G) renamed GLC
mkdir -p "$T/bad"
python3 -I - complex.cif "$T/bad/complex.cif" <<'PY'
import sys

out = []
for line in open(sys.argv[1]):
    if line.startswith("HETATM") and line.split()[9] == "G":  # column 9: label_asym_id
        line = line.replace(" MAN ", " GLC ")
    out.append(line)
open(sys.argv[2], "w").write("".join(out))
PY
if openmm relax "$T/bad/complex.cif" "$T/bad-out" 2> "$T/bad.err"; then
    echo "FAIL: relax accepted an unsupported sugar" >&2
    exit 1
fi

# two inputs sharing a stem
mkdir -p "$T/clash"
cp 1UBQ.pdb "$T/clash/a.pdb"
cp complex.cif "$T/clash/a.cif"
if openmm relax "$T/clash" "$T/clash-out" 2> "$T/clash.err"; then
    echo "FAIL: relax accepted two inputs sharing a stem" >&2
    exit 1
fi

# the input directory as the output, which would overwrite the inputs
if openmm relax "$T/in" "$T/in/." 2> "$T/inplace.err"; then
    echo "FAIL: relax accepted its input directory as the output" >&2
    exit 1
fi

# md: relax's GLYCAM-named output (the chaining path) for 0.2 ns, then extended to 0.4
openmm md "$T/relax/complex.cif" "$T/md" --ns 0.2 --seed 1
openmm md "$T/relax/complex.cif" "$T/md" --ns 0.4 --seed 1
hashes() { (cd "$T/md" && sha256sum -- *) > "$1"; }
hashes "$T/md-0.4.sha256"
# a smaller --ns: already there, so nothing changes
openmm md "$T/relax/complex.cif" "$T/md" --ns 0.3 2> "$T/md-smaller.err"
hashes "$T/md-0.3.sha256"

# a crash after a frame was written but before its state.xml was saved: one more
# frame (OpenMM's own DCDFile, appending) and one more log row than state.xml holds
python3 -I - "$T/md" <<'PY'
import sys
from pathlib import Path

from openmm.app import DCDFile, PDBFile

md = Path(sys.argv[1])
topology = PDBFile(str(md / "topology.pdb")).topology
final = PDBFile(str(md / "final.pdb"))
keep = [a.index for a in final.topology.atoms() if a.residue.name not in ("HOH", "NA", "CL")]
positions = final.getPositions(asNumpy=True)[keep]
with open(md / "trajectory.dcd", "r+b") as f:
    dcd = DCDFile(f, topology, 0.004, firstStep=25000, interval=25000, append=True)
    dcd.writeModel(positions, periodicBoxVectors=final.topology.getPeriodicBoxVectors())
row = (md / "log.csv").read_text().splitlines()[-1].split(",")
row[0], row[1] = str(int(row[0]) + 25000), str(float(row[1]) + 100)
with open(md / "log.csv", "a") as f:
    f.write(",".join(row) + "\n")
PY
openmm md "$T/relax/complex.cif" "$T/md" --ns 0.5 --seed 1
hashes "$T/md-0.5.sha256"

# a different input into the same directory
if openmm md "$T/in/1UBQ.pdb" "$T/md" --ns 0.5 2> "$T/md-other.err"; then
    echo "FAIL: md resumed a run of a different input" >&2
    exit 1
fi
hashes "$T/md-other.sha256"

# the skill's library-way script: md's last state run on for 2 ps in md's force field.
# This script is the skill's too (the skill's text is copied from here).
cat > "$T/continue.py" <<'PY'
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
PY
python3 -I "$T/continue.py" "$T/md" "$T/continue.dcd" 500 > "$T/continue.csv"

# ML/MM: AIMNet2 on the interface region of md's last output, the rest in the force field.
# This script is the skill's too (the skill's text is copied from here).
cat > "$T/mlmm.py" <<'PY'
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
    sys.exit(f"region charge {total:.3f} is not near an integer: cut the region at residue boundaries")

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
PY
python3 -I "$T/mlmm.py" "$T/md" "$T/mlmm" "A:20-21,A:24-24,A:28-28,A:31-31,A:40-40,A:44-44,A:46-47,A:50-51,B:4-5,B:8-8,B:11-12,B:15-16,B:19-20,B:22-23,B:26-26" > "$T/mlmm.txt"

# a span that matches no residue must stop and name the span
if python3 -I "$T/mlmm.py" "$T/md" "$T/mlmm-bad" "A:900-901" > /dev/null 2> "$T/mlmm-bad.err"; then
    echo "FAIL: mlmm.py accepted a span matching no residue" >&2
    exit 1
fi

python3 -I check.py "$T"
