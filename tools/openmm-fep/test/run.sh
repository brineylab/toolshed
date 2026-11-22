#!/usr/bin/env bash
# openmm-fep's test. Fixtures are openmm's test complex (tools/openmm/test/complex.cif, Boltz-2, two chains,
# glycans on A:17 and B:13) after `openmm relax`, as mmCIF and as PDB, made in the openmm 1.0.0 image with
# (openmm's input dir, not the output dir, since relax refuses an output that holds its inputs):
#   openmm relax /in/complex.cif /out/r                    -> complex.cif
#   python3 -I -c "from openmm.app import *; x = PDBxFile('/in/complex.cif'); PDBFile.writeFile(x.topology, x.positions, open('/o/complex.pdb', 'w'), keepIds=True)"
#   openmm relax /in/complex.pdb /out/r                    -> complex.pdb
# Runs, each at --cycles 1: A:L47A bound and unbound (--partner A); the bound one again, which must be a no-op;
# the unbound one's result.json deleted, which must be redone; A:K44A bound (charge-changing, so the alchemical
# water); A:G38A bound (Gly to Ala, whose CB is built); complex.pdb unbound; then the refusals, each checked
# for exit 2 and its stderr word, writing nothing; then analyze over the L47A pair and its refusals. check.py verifies the answers. Writes only under /tmp.
# With FLYLOFT_OUT set, every case's output tree is copied there at the end.
set -euo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d /tmp/openmm-fep.XXXXXX)
trap 'rm -rf "$T"' EXIT

python3 -I check.py mapping complex.cif

timed() {  # timed NAME COMMAND...: the run's wall time on stderr
    local start=$SECONDS name=$1
    shift
    "$@"
    echo "case $name: $((SECONDS - start)) s" >&2
}
tree() { (cd "$1" && find . -type f | sort | xargs sha256sum); }

# 1, 2: L47A, both legs
timed bound openmm-fep run complex.cif "$T/bound" --mutation A:L47A --leg bound --replicate 0 --cycles 1
timed unbound openmm-fep run complex.cif "$T/unbound" --mutation A:L47A --leg unbound --partner A --replicate 0 --cycles 1

# 3: a finished replicate is a no-op
R1="$T/bound/bound/replicate-0"
before=$(sha256sum "$R1/result.json")
openmm-fep run complex.cif "$T/bound" --mutation A:L47A --leg bound --replicate 0 --cycles 1 2>"$T/err"
grep -q "already done" "$T/err"
[ "$before" = "$(sha256sum "$R1/result.json")" ]

# 4: a crashed replicate is redone
R2="$T/unbound/unbound/replicate-0"
rm "$R2/result.json"
timed redo openmm-fep run complex.cif "$T/unbound" --mutation A:L47A --leg unbound --partner A --replicate 0 --cycles 1 2>"$T/err"
grep -q "starting it again" "$T/err"
test -s "$R2/result.json"

# 5, 6, 7: K44A (charge-changing), G38A (Gly to Ala), and the PDB input
timed k44a openmm-fep run complex.cif "$T/k44a" --mutation A:K44A --leg bound --replicate 0 --cycles 1
timed g38a openmm-fep run complex.cif "$T/g38a" --mutation A:G38A --leg bound --replicate 0 --cycles 1
timed pdb openmm-fep run complex.pdb "$T/pdb" --mutation A:L47A --leg unbound --partner A --replicate 0 --cycles 1

python3 -I check.py run "$T"

# 8: refusals, exit 2, the problem named, nothing written
refused() {  # refused WORD COMMAND...
    local word=$1 rc=0
    shift
    "$@" 2>"$T/err" || rc=$?
    [ "$rc" = 2 ] && grep -q -- "$word" "$T/err" || { echo "expected exit 2 naming '$word', got $rc: $(cat "$T/err")" >&2; exit 1; }
}
for case in "wild-type|A:F47A|bound" "glycan|A:N17A|bound" "not in the structure|A:L47aA|bound" "itself|A:L47L|bound"; do
    IFS='|' read -r word mutation leg <<<"$case"
    refused "$word" openmm-fep run complex.cif "$T/refused" --mutation "$mutation" --leg "$leg" --replicate 0
done
refused "mutated chain" openmm-fep run complex.cif "$T/refused" --mutation A:L47A --leg unbound --partner B --replicate 0
refused "--partner" openmm-fep run complex.cif "$T/refused" --mutation A:L47A --leg bound --partner A --replicate 0
refused "--partner" openmm-fep run complex.cif "$T/refused" --mutation A:L47A --leg unbound --replicate 0
test ! -e "$T/refused"
# an output directory run with another mutation: refused, its files untouched
before=$(tree "$T/bound")
refused "mutation" openmm-fep run complex.cif "$T/bound" --mutation A:L47G --leg bound --replicate 0 --cycles 1
[ "$before" = "$(tree "$T/bound")" ]

# 9: analyze pools each leg: three rows, ddg = bound - unbound; then its refusals
mkdir -p "$T/pool" "$T/mix/bound" "$T/nores/unbound"
cp -r "$T/bound/bound" "$T/unbound/unbound" "$T/pool"
openmm-fep analyze "$T/pool" "$T/ddg.tsv"
python3 -I check.py analyze "$T/ddg.tsv"
cp -r "$T/unbound/unbound" "$T/mix"
cp -r "$T/bound/bound/replicate-0" "$T/mix/bound/replicate-0"
cp -r "$T/g38a/bound/replicate-0" "$T/mix/bound/replicate-1"
refused "mutation" openmm-fep analyze "$T/mix" "$T/mix.tsv"
refused "unbound" openmm-fep analyze "$T/k44a" "$T/k44a.tsv"
cp -r "$T/bound/bound" "$T/nores"
cp -r "$T/unbound/unbound/replicate-0" "$T/nores/unbound/replicate-0"
rm "$T/nores/unbound/replicate-0/result.json"
refused "unbound" openmm-fep analyze "$T/nores" "$T/nores.tsv"
test ! -e "$T/mix.tsv" -a ! -e "$T/k44a.tsv" -a ! -e "$T/nores.tsv"

if [ -n "${FLYLOFT_OUT:-}" ]; then
    mkdir -p "$FLYLOFT_OUT"
    for d in bound unbound k44a g38a pdb; do cp -r "$T/$d" "$FLYLOFT_OUT/$d"; done
fi
echo "openmm-fep test passed" >&2
