"""Fixture facts and checks for the openmm test (run as `python3 -I check.py <tmpdir>`).

complex.cif is Boltz-2's model_0 for complex.yaml, unmodified (CPU, seed 0, single-sequence).
It has no _struct_conn loop, so no glycosidic or Asn-NAG bond is recorded: the test must find
them by distance; run.sh writes a copy with the six links in a _struct_conn loop for the bonded
path, and another with the core's sugars in one chain as AlphaFold 3 writes them. Boltz writes each sugar as its own chain (C-H), HETATM, residue number 1.

Asn sites, as (chain, residue number): the lone NAG (chain C) on A:17, the core on B:13.
Core, from the Asn: D NAG, E NAG, F BMA, G MAN (alpha1-3), H MAN (alpha1-6).
"""

ASN_SITES = [("A", 17), ("B", 13)]

# whole residues of both chains with a heavy atom within 4.5 A of the other chain, computed from
# complex.cif; the termini A53 and B28 are left out, and so is K35, so the region has
# a nonzero net charge and the test can tell that the charge reaches AIMNet2 (23 residues)
ML_REGION = ("A:20-21,A:24-24,A:28-28,A:31-31,A:40-40,A:44-44,A:46-47,A:50-51,"
             "B:4-5,B:8-8,B:11-12,B:15-16,B:19-20,B:22-23,B:26-26")

# Asp/Glu -1, Lys/Arg +1, His neutral at pH 7 (none present), counted from the residue types:
# A: E21 K28 K44 E51 = 0; B: E4 E11 R22 = -1; caps are neutral
ML_REGION_CHARGE = -1

# aimnet 0.2.6 calculators/calculator.py:209 `nb_threshold: int = 120,` and :1932
# `or self.nb_threshold < N`: more than 120 atoms switches from dense to neighbor list.
# The region (23 residues, about 17 atoms each with hydrogens, about 400 atoms) is above it.
AIMNET_NB_THRESHOLD = 120
# the region's atom count, pinned from the first passing run
ML_REGION_ATOMS = 400
ML_LOG_ROWS = 10  # N_STEPS 1000 logged every 100 steps

# relax: GLYCAM residue names the fixture must come out with. Asn carrying a sugar is NLN; the
# sugars, in chain order C-H: the lone GlcNAc is terminal beta-GlcNAc (0YB); the core's two
# GlcNAc are 4-linked (4YB), the beta-Man is 3,6-linked (VMB), the two alpha-Man are terminal
# (0MA). The codes follow GLYCAM's naming guide (glycam.org/docs/forcefield/glycam-naming-2,
# Tables 1, 2 and 4) and each is a template in OpenMM's amber14/GLYCAM_06j-1.xml.
SUGAR_CHAINS = ["C", "D", "E", "F", "G", "H"]
SUGAR_NAMES = ["0YB", "4YB", "4YB", "VMB", "0MA", "0MA"]
RELAX_HEADER = "name\tenergy_before\tenergy_after\theavy_atom_rmsd"
RELAX_NAMES = ["1UBQ.pdb", "complex.cif"]
RELAX_MAX_RMSD = 0.5  # Angstrom; the restraints are meant to allow a few tenths
# relax/complex.cif's atom count, pinned from the first passing run (2026-11-20, CPU, the
# dev image): both chains and all six sugars with hydrogens at pH 7, each linked sugar's
# leaving O1 gone. A dropped or doubled hydrogen, or a kept O1, changes it. The complex has
# no His, so no protonation choice moves it.
COMPLEX_ATOMS = 1478


# Glycan links found by distance, one count per structure in input order: the directory run
# is 1UBQ.pdb (no glycans, 0) then complex.cif (no _struct_conn, all 6); the _struct_conn
# variants record all 6, so none is inferred and the bonded path is the one exercised.
INFERRED_PLAIN = [0, 6]
INFERRED_WITHBONDS = [0]
INFERRED_AF3 = [0]

# md: frames every 25,000 steps (100 ps at 4 fs), counted from production's step 0. The test
# ends at 0.5 ns, so 5 frames and log rows at 25,000 ... 125,000, each step once.
MD_INTERVAL = 25000
MD_STEPS = [MD_INTERVAL * k for k in range(1, 6)]
SOLVENT = {"HOH", "NA", "CL"}
MIN_BOX_SIDE = 30.0  # Angstrom: cuPotential needs three 1.0 nm cutoffs per side
# OpenMM's DCDFile (openmm/app/dcdfile.py, 8.6.1): a 276-byte header (frame count at byte 8,
# first step at 12, interval at 16, last frame's step at 20, atom count at 268), then per frame
# a 56-byte box record and three 4N-byte coordinate blocks, each between two 4-byte lengths.
DCD_HEADER = 276
# continue.py: 500 steps from md's state.xml, a frame and a stdout row every 250 steps (1 ps).
# Its potential must stay within 1% of md's last logged one (same force field and system) and
# its temperature near 300 K (the velocities came from the state; from rest it would be far lower).
CONTINUE_STEPS = [250, 500]
CONTINUE_POTENTIAL_TOL = 0.01
CONTINUE_T = (280.0, 320.0)


def fail(msg):
    raise SystemExit(f"FAIL: {msg}")


def residues(path):
    from openmm.app import PDBxFile

    return list(PDBxFile(str(path)).topology.residues())


def check_relax_tsv(path):
    lines = path.read_text().splitlines()
    if lines[0] != RELAX_HEADER:
        fail(f"{path}: header {lines[0]!r}")
    rows = [line.split("\t") for line in lines[1:]]
    if [row[0] for row in rows] != RELAX_NAMES:
        fail(f"{path}: rows {[row[0] for row in rows]}, expected {RELAX_NAMES}")
    for name, before, after, rmsd in rows:
        if not float(after) < float(before):
            fail(f"{name}: energy rose, {before} -> {after}")
        if not float(rmsd) < RELAX_MAX_RMSD:
            fail(f"{name}: heavy-atom RMSD {rmsd} >= {RELAX_MAX_RMSD}")


def check_complex(path):
    res = residues(path)
    by_site = {(r.chain.id, int(r.id)): r.name for r in res}
    for site in ASN_SITES:
        if by_site.get(site) != "NLN":
            fail(f"{path}: {site} is {by_site.get(site)}, expected NLN")
    sugars = [r for r in res if r.chain.id in SUGAR_CHAINS]
    if [r.name for r in sugars] != SUGAR_NAMES:
        fail(f"{path}: sugars {[r.name for r in sugars]}, expected {SUGAR_NAMES}")
    for r in sugars:
        if not any(a.element.symbol == "H" for a in r.atoms()):
            fail(f"{path}: sugar {r.name} chain {r.chain.id} has no hydrogens")
    natoms = sum(len(list(r.atoms())) for r in res)
    if natoms != COMPLEX_ATOMS:
        fail(f"{path}: {natoms} atoms, expected {COMPLEX_ATOMS}")
    return [r.name for r in res]


def main():
    from pathlib import Path
    import re
    import sys

    T = Path(sys.argv[1])
    check_relax_tsv(T / "relax" / "relax.tsv")
    names = check_complex(T / "relax" / "complex.cif")
    if check_complex(T / "relax-withbonds" / "complex.cif") != names:
        fail("the _struct_conn variant's residue names differ from the distance-inferred ones")
    if check_complex(T / "relax-af3" / "complex.cif") != names:
        fail("the AlphaFold 3 layout's residue names differ from the distance-inferred ones")
    if check_complex(T / "rerelax" / "complex.cif") != names:
        fail("relaxing relax's own output changed its residue names")
    for err, expected in (("relax.err", INFERRED_PLAIN), ("withbonds.err", INFERRED_WITHBONDS),
                          ("af3.err", INFERRED_AF3)):
        found = [int(n) for n in re.findall(r"inferred (\d+) glycan link", (T / err).read_text())]
        if found != expected:
            fail(f"{err}: links inferred by distance {found}, expected {expected}")
    err = (T / "bad.err").read_text()
    if "complex.cif: unsupported residue GLC chain G" not in err:
        fail(f"the unsupported sugar's error does not name the file and GLC chain G: {err!r}")
    out = T / "clash-out"
    if out.exists() and any(p.suffix in (".pdb", ".cif") for p in out.iterdir()):
        fail("the stem clash wrote a structure")
    if "a.cif" not in (T / "clash.err").read_text():
        fail("the stem clash's error does not name the files")
    if sorted(p.name for p in (T / "in").iterdir()) != ["1UBQ.pdb", "complex.cif"] \
            or any((T / "in" / n).read_bytes() != Path(n).read_bytes() for n in ("1UBQ.pdb", "complex.cif")):
        fail("relax into its input directory changed the directory")
    if "input" not in (T / "inplace.err").read_text():
        fail(f"relax into its input directory does not say why: {(T / 'inplace.err').read_text()!r}")
    check_md(T)
    check_continue(T)
    check_mlmm(T)
    print(f"openmm test: relax passed ({len(names)} residues in the complex, GLYCAM names on both paths)")


def check_md(T):
    import hashlib
    import shutil
    import struct

    from openmm.app import DCDFile, PDBFile

    md = T / "md"
    if (T / "md-0.3.sha256").read_text() != (T / "md-0.4.sha256").read_text():
        fail("md with a smaller --ns changed its output directory")
    if not (T / "md-smaller.err").read_text().strip():
        fail("md with a smaller --ns said nothing on stderr")
    if (T / "md-other.sha256").read_text() != (T / "md-0.5.sha256").read_text():
        fail("md with a different input changed the output directory")
    if "input.sha256" not in (T / "md-other.err").read_text():
        fail(f"md with a different input does not say why: {(T / 'md-other.err').read_text()!r}")
    if (md / "input.sha256").read_text().split()[0] != hashlib.sha256((T / "relax" / "complex.cif").read_bytes()).hexdigest():
        fail("input.sha256 is not the input's sha256")
    for name in ("system.pdb", "equilibrated.pdb", "final.pdb", "state.xml"):
        if not (md / name).stat().st_size:
            fail(f"md wrote no {name}")

    sides = [float(x) for x in (md / "system.pdb").read_text().split("CRYST1", 1)[1].split()[:3]]
    if min(sides) < MIN_BOX_SIDE:
        fail(f"system.pdb's box {sides} A has a side under {MIN_BOX_SIDE} A")

    topology = PDBFile(str(md / "topology.pdb")).topology
    found = {r.name for r in topology.residues()} & SOLVENT
    if found:
        fail(f"topology.pdb holds solvent: {sorted(found)}")
    if topology.getNumAtoms() != COMPLEX_ATOMS:
        fail(f"topology.pdb has {topology.getNumAtoms()} atoms, expected the complex's {COMPLEX_ATOMS}")

    data = (md / "trajectory.dcd").read_bytes()
    frames, first, interval, last = struct.unpack("<4i", data[8:24])
    natoms = struct.unpack("<i", data[268:272])[0]
    frame = 56 + 3 * (8 + 4 * natoms)
    if natoms != topology.getNumAtoms():
        fail(f"trajectory.dcd has {natoms} atoms per frame, topology.pdb {topology.getNumAtoms()}")
    if (frames, first, interval, last) != (len(MD_STEPS), MD_INTERVAL, MD_INTERVAL, MD_STEPS[-1]):
        fail(f"trajectory.dcd's header: {frames} frames, steps {first}+{interval}k, last {last}")
    if len(data) != DCD_HEADER + len(MD_STEPS) * frame:
        fail(f"trajectory.dcd is {len(data)} bytes, expected {DCD_HEADER} + {len(MD_STEPS)} x {frame}")
    copy = T / "check.dcd"
    shutil.copy(md / "trajectory.dcd", copy)
    with open(copy, "r+b") as f:  # OpenMM's own reader of the header: magic and atom count
        DCDFile(f, topology, 0.004, append=True)

    lines = (md / "log.csv").read_text().splitlines()
    if not lines[0].startswith("#"):
        fail(f"log.csv's first line is not a header: {lines[0]!r}")
    steps = [int(line.split(",")[0]) for line in lines[1:]]
    if steps != MD_STEPS:
        fail(f"log.csv's steps {steps}, expected {MD_STEPS}")
    print(f"openmm test: md passed ({natoms} atoms in {len(MD_STEPS)} frames, box {sides} A)")


def check_continue(T):
    import math
    import struct

    from openmm.app import PDBFile

    natoms = PDBFile(str(T / "md" / "system.pdb")).topology.getNumAtoms()
    data = (T / "continue.dcd").read_bytes()
    frames, size = struct.unpack("<i", data[8:12])[0], struct.unpack("<i", data[268:272])[0]
    if (frames, size) != (len(CONTINUE_STEPS), natoms):
        fail(f"continue.dcd: {frames} frames of {size} atoms, expected {len(CONTINUE_STEPS)} of system.pdb's {natoms}")
    if len(data) != DCD_HEADER + frames * (56 + 3 * (8 + 4 * natoms)):
        fail(f"continue.dcd is {len(data)} bytes, not {frames} whole frames")
    lines = (T / "continue.csv").read_text().splitlines()
    rows = [[float(x) for x in line.split(",")] for line in lines[1:]]
    if not lines[0].startswith("#") or [int(r[0]) for r in rows] != CONTINUE_STEPS:
        fail(f"continue.py's stdout: {lines!r}")
    md_last = float((T / "md" / "log.csv").read_text().splitlines()[-1].split(",")[2])
    for step, potential, temperature in rows:
        if not (math.isfinite(potential) and abs(potential - md_last) < CONTINUE_POTENTIAL_TOL * abs(md_last)):
            fail(f"continue.py at step {step:.0f}: potential {potential}, md's last {md_last}")
        if not CONTINUE_T[0] < temperature < CONTINUE_T[1]:
            fail(f"continue.py at step {step:.0f}: temperature {temperature} K")
    print(f"openmm test: continue passed ({frames} frames of {natoms} atoms, potential {rows[-1][1]:.0f} kJ/mol)")


def check_mlmm(T):
    import math

    out = dict(line.split() for line in (T / "mlmm.txt").read_text().splitlines())
    if int(out["region_atoms"]) != ML_REGION_ATOMS or ML_REGION_ATOMS <= AIMNET_NB_THRESHOLD:
        fail(f"region_atoms {out['region_atoms']}, expected {ML_REGION_ATOMS} (> {AIMNET_NB_THRESHOLD})")
    if int(out["region_charge"]) != ML_REGION_CHARGE:
        fail(f"region_charge {out['region_charge']}, expected {ML_REGION_CHARGE}")
    rows = (T / "mlmm" / "log.csv").read_text().splitlines()[1:]
    if len(rows) != ML_LOG_ROWS or not all(math.isfinite(float(r.split(",")[1])) for r in rows):
        fail(f"mlmm log.csv has {len(rows)} rows (expected {ML_LOG_ROWS}) or a non-finite energy")
    if "A:900-901" not in (T / "mlmm-bad.err").read_text():
        fail("mlmm.py's error for a span matching no residue does not name the span")
    if not (T / "mlmm" / "final.pdb").stat().st_size:
        fail("mlmm wrote no final.pdb")
    print(f"openmm test: mlmm passed ({out['region_atoms']} ML atoms, charge {out['region_charge']})")


if __name__ == "__main__":
    main()
