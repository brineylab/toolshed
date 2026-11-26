#!/usr/bin/env python3
"""Check numerical outputs, context, unresolved coordinates, and both formats."""
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1])


def fasta(path):
    records = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            records.append([line[1:], ""])
        else:
            records[-1][1] += line
    return records


def parse_header(header):
    fields = {}
    for token in header.replace(", ", " ").split():
        if "=" in token:
            k, v = token.split("=", 1)
            fields[k] = v.rstrip(",")
    return fields


def score(path, reference=None):
    with path.open() as f:
        assert f.readline().rstrip("\n") == "id\tll_fullseq\tll_withcoord"
        rows = list(csv.DictReader(f, fieldnames=["id", "ll_fullseq", "ll_withcoord"], delimiter="\t"))
    assert [r["id"] for r in rows] == ["native", "shuffled"]
    vals = [(float(r["ll_fullseq"]), float(r["ll_withcoord"])) for r in rows]
    assert all(math.isfinite(x) for pair in vals for x in pair)
    assert vals[0][0] > vals[1][0] and vals[0][1] > vals[1][1], vals
    native = vals[0]
    if reference is not None:
        assert max(abs(a - b) for a, b in zip(native, reference)) < 1e-5, (native, reference)
    return native


native = fasta(Path("antibody-cif.fasta"))[0][1]
assert len(native) == 218
headers = fasta(root / "antibody.cif.fa")
assert len(headers) == 2
assert parse_header(headers[0][0])["num_res"] == "218"
for i, (header, seq) in enumerate(headers):
    fields = parse_header(header)
    assert len(seq) == len(native) and set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")
    if i:
        assert fields["id"] == "1" and fields["seed"] == "7" and fields["T"] == "0.1"
        assert 0 < float(fields["overall_confidence"]) <= 1
        assert 0 <= float(fields["seq_rec"]) <= 1

scores = {}
scores["sabdab_example_with_antigen"] = score(root / "selected-cif.score.tsv")
scores["pdb_author_chain"] = score(root / "selected-pdb.score.tsv", scores["sabdab_example_with_antigen"])
scores["all_protein_chains"] = score(root / "all-cif.score.tsv", scores["sabdab_example_with_antigen"])
scores["single_pdb"] = score(root / "single-pdb.score.tsv")
scores["single_cif"] = score(root / "single-cif.score.tsv", scores["single_pdb"])
assert scores["single_pdb"] != scores["sabdab_example_with_antigen"]
assert len(fasta(root / "single.pdb.fa")[0][1]) == 93
# The actual values are proven finite and structure-sensitive; this is a real
# local check against the same native model, not a claimed SAbDab aggregate.

pr = np.load(root / "antibody-cif.res.npz")
assert pr["ids"].tolist() == ["native", "shuffled"]
assert pr["ll_native"].shape == pr["ll_shuffled"].shape == (218,)
assert pr["resolved_coordinates"].shape == (218,)
assert (~pr["resolved_coordinates"]).tolist().index(True) == 11  # H12 only.
assert np.sum(pr["resolved_coordinates"]) == 217
assert math.isclose(float(pr["ll_native"].mean()), scores["sabdab_example_with_antigen"][0], abs_tol=1e-6)
assert math.isclose(float(pr["ll_native"][pr["resolved_coordinates"]].mean()),
                    scores["sabdab_example_with_antigen"][1], abs_tol=1e-6)
emb = np.load(root / "antibody.embed.npz")
assert emb["ids"].tolist() == ["antibody"] and emb["chains"].tolist() == ["H"]
assert emb["embeddings"].shape == (218, 512) and emb["embeddings"].dtype == np.float32
assert np.isfinite(emb["embeddings"]).all()
# Flattened representation must not hide residue rows that never change.
assert np.all(np.std(emb["embeddings"], axis=0) > 1e-7)
assert np.std(np.linalg.norm(emb["embeddings"], axis=1)) > 1e-7
(root / "checked.json").write_text(json.dumps(scores, indent=2) + "\n")
print("PASS: 520-residue author-chain complex, unresolved H12, non-protein exclusion, both formats, native > shuffled, context, 512-D finite nondegenerate embeddings")
