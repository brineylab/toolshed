"""Fixture facts and checks for the openmm test (run as `python3 -I check.py <tmpdir>`).

complex.cif is Boltz-2's model_0 for complex.yaml, unmodified (CPU, seed 0, single-sequence).
It has no _struct_conn loop, so no glycosidic or Asn-NAG bond is recorded: the test must find
them by distance; run.sh writes a copy with the six links in a _struct_conn loop for the bonded
path. Boltz writes each sugar as its own chain (C-H), HETATM, residue number 1.

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
# variant records all 6, so none is inferred and the bonded path is the one exercised.
INFERRED_PLAIN = [0, 6]
INFERRED_WITHBONDS = [0]


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
    if check_complex(T / "rerelax" / "complex.cif") != names:
        fail("relaxing relax's own output changed its residue names")
    for err, expected in (("relax.err", INFERRED_PLAIN), ("withbonds.err", INFERRED_WITHBONDS)):
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
    print(f"openmm test: relax passed ({len(names)} residues in the complex, GLYCAM names on both paths)")


if __name__ == "__main__":
    main()
