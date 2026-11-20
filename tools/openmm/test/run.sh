#!/usr/bin/env bash
# openmm's test. Fixtures: complex.cif (Boltz-2, two chains, a branched
# N-glycan core on B:13 and a lone GlcNAc on A:17, no _struct_conn, so the
# glycan links are found by distance) and water-stripped 1UBQ. Runs:
# `openmm relax` on a directory of both; the complex again with its six links
# written into a _struct_conn loop, which must give the same GLYCAM names;
# the complex with its first MAN renamed GLC, which must stop naming GLC; and
# a directory where a.pdb and a.cif share a stem, which must stop before
# writing anything. check.py verifies the answers. Writes only under /tmp.
set -euo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d /tmp/openmm.XXXXXX)
trap 'rm -rf "$T"' EXIT

mkdir -p "$T/in"
cp complex.cif 1UBQ.pdb "$T/in/"
openmm relax "$T/in" "$T/relax"

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
openmm relax "$T/withbonds/complex.cif" "$T/relax-withbonds"

# an unsupported sugar: the first MAN (chain G) renamed GLC
mkdir -p "$T/bad"
python3 -I - complex.cif "$T/bad/complex.cif" <<'PY'
import sys

out = []
for line in open(sys.argv[1]):
    if line.startswith("HETATM") and line.split()[9] == "G":
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

python3 -I check.py "$T"
