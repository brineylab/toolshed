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
    # ANARCI's CSV uses encounter ranks, not alphabetical insertion order:
    # in IMGT CDR3 it preserves the special 111/112 ordering exactly by removal.
    stripped = [p.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for p in positions]
    assert [int(p) for p in stripped] == sorted(int(p) for p in stripped)
    assert stripped.index("111") + 1 == stripped.index("112")
    # Do not normalize labels lexicographically: consumers must retain this order.
    assert any(p.startswith("111") and p != "111" for p in positions)
    assert any(p.startswith("112") and p != "112" for p in positions)
    rows = list(reader)
assert len(rows) == len(seqs) == 4, (len(rows), len(seqs))
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
