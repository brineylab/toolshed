#!/usr/bin/env python3
"""Check LigandMPNN's test outputs against the known fixture.

1BC8's chain C is 93 residues, author-numbered 1-93, starting MDSAI; its
ligand context is two DNA chains and two zinc ions. The native record's
header carries num_res and a positive num_ligand_res; every design is 93
amino acids; the five fixed residues stay MDSAI; stats and packed outputs
are real. num_res counts the residues the model actually samples —
torch.sum(rec_mask) over mask * chain_mask in upstream run.py — so it is
93 when the whole chain is designed and 88 in the fixed run (93 minus the
five fixed residues C1-C5), not 93 everywhere.
"""
import re
import sys
from pathlib import Path

T = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
AA = set("ACDEFGHIKLMNPQRSTVWY")
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def fail(msg):
    raise SystemExit(f"FAIL: {msg}")


def records(fa_path):
    """Each record is one '>' header line followed by one sequence line."""
    lines = fa_path.read_text().splitlines()
    recs = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            fail(f"{fa_path}: line {i+1} does not start a record")
        if i + 1 >= len(lines) or not lines[i + 1]:
            fail(f"{fa_path}: record at line {i+1} has no sequence")
        recs.append((lines[i], lines[i + 1]))
        i += 2
    return recs


def native_pdb_sequence(pdb_path):
    seq, seen = [], set()
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16] == " CA " and line[21] == "C":
            key = (line[21], line[22:27])
            if key in seen:
                fail(f"{pdb_path}: chain C residue {key} appears twice")
            seen.add(key)
            seq.append(THREE_TO_ONE[line[17:20].strip()])
    return "".join(seq)


def check_designs(fa_path, expect_native, n_designs, tag, expect_num_res):
    recs = records(fa_path)
    if len(recs) != n_designs + 1:
        fail(f"{fa_path}: expected {n_designs + 1} records, got {len(recs)}")
    head, seq = recs[0]
    if seq != expect_native:
        fail(f"{fa_path}: native record does not match the PDB chain C sequence")
    m = re.search(r"num_res=(\d+)", head)
    if not m or int(m.group(1)) != expect_num_res:
        fail(f"{fa_path}: native header num_res missing or wrong "
             f"(expected {expect_num_res} sampled residues): {head}")
    m = re.search(r"num_ligand_res=(\d+)", head)
    if not m or int(m.group(1)) <= 0:
        fail(f"{fa_path}: native header num_ligand_res missing or zero: {head}")
    if "use_ligand_context=True" not in head:
        fail(f"{fa_path}: ligand context not on: {head}")
    for k, (head, seq) in enumerate(recs[1:], start=1):
        m = re.search(r"id=(\d+)", head)
        if not m or int(m.group(1)) != k:
            fail(f"{fa_path}: design {k} header id wrong: {head}")
        for field in ("overall_confidence", "ligand_confidence", "seq_rec"):
            m = re.search(field + r"=([0-9.]+)", head)
            if not m:
                fail(f"{fa_path}: design {k} missing {field}: {head}")
            v = float(m.group(1))
            if not 0.0 < v <= 1.0:
                fail(f"{fa_path}: design {k} {field}={v} out of (0, 1]")
        m = re.search(r"overall_confidence=([0-9.]+)", head)
        if float(m.group(1)) < 0.05:
            fail(f"{fa_path}: design {k} overall_confidence suspiciously low: {head}")
        m = re.search(r"seq_rec=([0-9.]+)", head)
        if float(m.group(1)) < 0.25:
            fail(f"{fa_path}: design {k} seq_rec suspiciously low: {head}")
        if len(seq) != 93 or set(seq) - AA:
            fail(f"{fa_path}: design {k} sequence not 93 amino acids: {seq[:20]}...")
    return [seq for _, seq in recs[1:]]


def ca_count(pdb_path):
    n = 0
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16] == " CA " and line[21] == "C":
            n += 1
    return n


native = native_pdb_sequence(HERE / "1BC8.pdb")
if not native.startswith("MDSAI") or len(native) != 93:
    fail(f"fixture 1BC8.pdb chain C is not the expected 93-residue MDSAI... domain")

designs = check_designs(T / "basic" / "seqs" / "1BC8.fa", native, 32, "basic", 93)
bb = T / "basic" / "backbones" / "1BC8_32.pdb"
if not bb.is_file():
    fail("basic/backbones/1BC8_32.pdb missing")
if ca_count(bb) != 93:
    fail(f"basic/backbones/1BC8_32.pdb has {ca_count(bb)} chain C CA atoms, not 93")

designs = check_designs(T / "fixed" / "seqs" / "1BC8.fa", native, 8, "fixed", 88)
for k, seq in enumerate(designs, start=1):
    if seq[:5] != "MDSAI":
        fail(f"fixed run design {k} broke fixed residues C1-C5: {seq[:10]}...")

import torch  # the image's python carries torch

stats = torch.load(T / "fixed" / "stats" / "1BC8.pt", map_location="cpu")
if not isinstance(stats, dict) or "generated_sequences" not in stats:
    fail("fixed/stats/1BC8.pt is not a stats dict with generated_sequences")
shape = tuple(stats["generated_sequences"].shape)
if shape != (8, 93):
    fail(f"fixed/stats/1BC8.pt generated_sequences shape {shape}, not (8, 93)")

packed = T / "fixed" / "packed" / "1BC8_packed_8_4.pdb"
if not packed.is_file():
    fail("fixed/packed/1BC8_packed_8_4.pdb missing")
if ca_count(packed) != 93:
    fail(f"fixed/packed/1BC8_packed_8_4.pdb has {ca_count(packed)} chain C CA atoms, not 93")

check_designs(T / "multi" / "seqs" / "1BC8.fa", native, 4, "multi", 93)

print("PASS: ligandmpnn designed 1BC8 chain C with DNA/zinc context; 32 + 8 + 4")
print("PASS: sequences are 93 amino acids; confidences and seq_rec in range;")
print("PASS: fixed residues C1-C5 kept as MDSAI; stats and packed PDBs written.")
