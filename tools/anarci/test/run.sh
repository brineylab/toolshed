#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
work=$(mktemp -d /tmp/anarci-test.XXXXXX)
trap 'rm -rf "$work"' EXIT

hmmscan -h | head -n 4 | grep -F '# HMMER 3.3.2'
python3 -I -c 'import importlib.metadata as m; assert m.version("anarci") == "2026.2.13.2"'
# This is the full pipeline used by epam-pcps, including process-pool numbering.
ANARCI -i fixture/heavy.fasta -o "$work/numbered_imgt" \
  -s imgt -r heavy --use_species human --csv -p 2 --assign_germline
[ -s "$work/numbered_imgt_H.csv" ]
[ ! -e "$work/numbered_imgt_KL.csv" ]
python3 -I - "$work/numbered_imgt_H.csv" fixture/heavy.fasta <<'PY'
import csv
import importlib.resources
import sys

path, fasta = sys.argv[1:]
seqs = {}
with open(fasta) as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
            # The fourth record is a negative control: no antibody domain.
            if name != "0|149198|not-a-heavy-chain":
                seqs[name] = ""
        elif line and name in seqs:
            seqs[name] += line
with open(path, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    assert header[:13] == [
        "Id", "domain_no", "hmm_species", "chain_type", "e-value", "score",
        "seqstart_index", "seqend_index", "identity_species", "v_gene",
        "v_identity", "j_gene", "j_identity",
    ], header[:13]
    positions = header[13:]
    # ANARCI emits CDR3 insertions in structural order about 111/112:
    # 111 letters ascend, while 112 letters descend. Do not numerically sort them.
    assert positions.index("111") < positions.index("112")
    assert "111A" in positions and "112A" in positions
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    insertions = {}
    for number in ["111", "112"]:
        subset = [p for p in positions if p.startswith(number) and p != number]
        assert all(len(p) == len(number) + 1 and p[-1] in letters for p in subset)
        insertions[number] = [p[-1] for p in subset]
    assert insertions["111"] == sorted(insertions["111"])
    assert insertions["112"] == sorted(insertions["112"], reverse=True)
    rows = list(reader)
assert len(rows) == len(seqs) == 3, (len(rows), len(seqs))
seen = set()
for row in rows:
    assert len(row) == len(header)
    sample_id, family, name = row[0].split("|")
    assert sample_id == "0" and family == "149198"
    seen.add(row[0])
    assert row[1] == "0" and row[3] == "H"
    assert row[8] == "human", row[:13]
    assert row[9].startswith(("human_", "IGHV")) or "IGHV" in row[9], row[9]
    assert row[11].startswith(("human_", "IGHJ")) or "IGHJ" in row[11], row[11]
    observed = "".join("" if a == "-" else a for a in row[13:])
    assert observed == seqs[row[0]], (row[0], observed, seqs[row[0]])
    assert row[13 + positions.index("118")] == "W"
assert seen == set(seqs)
assert importlib.resources.files("anarci").joinpath("dat/HMMs/ALL.hmm").is_file()
print(f"ANARCI: {len(rows)} heavy chains; {len(positions)} ordered IMGT columns; exact sequences reconstructed")
PY
