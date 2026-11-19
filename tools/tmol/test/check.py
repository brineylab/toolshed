#!/usr/bin/env python3
"""Check tmol's test outputs against the known fixture.

The fixture is water-stripped 1UBQ (76 residues; the record's 58 waters
removed) in 26 copies. 26 crosses tmol's first input-growth threshold:
the packer chunks a pose stack whose poses hold at most 256 blocks into
chunks of 25 poses, so `tmol pack` on this batch runs the chunked path --
25 poses, then 1. Fewer than 25 poses would never leave one chunk.

The reference values are pinned from the campaign's measurement run; the
docstring below records where they came from and what the tolerances
cover. TODO(measure): fill in the run, the hardware, and the rationale
for each tolerance.
"""
import csv
import math
import sys
from pathlib import Path

T = Path(sys.argv[1])

NPOSES = 26
NAMES = [f"pose_{i:02d}.pdb" for i in range(1, NPOSES + 1)]

# TODO(measure): pinned by the measurement run.
HEADER = None  # the exact "name\ttotal\t<terms>" header of a score TSV
REFERENCE_TOTAL = None  # 1UBQ's beta2016 total at seed 0
TOTAL_TOLERANCE = None  # slack on the reference, across processes and GPUs
POSE_SPREAD = None  # identical poses in one batch may still spread by this
MIN_LOWER_BY = None  # minimize lowers every pose by at least this
PACK_LOWER_BY = None  # pack likewise
RELAX_LOWER_BY = None  # relax likewise
DEMO_LOWER_BY = None  # the library script's relax lowers by at least this
SEEDED_MAX_TOTAL_DELTA = None  # seeded pack pair: None = byte-identical
SEEDED_MAX_COORD_DELTA = None  # seeded pack pair: max coordinate delta, A
CHAIN_EXACT = True  # tmol score <outdir> reproduces scores.tsv byte for byte


def fail(msg):
    raise SystemExit(f"FAIL: {msg}")


def rows(tsv):
    with open(tsv) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def totals_of(tsv):
    return [float(row["total"]) for row in rows(tsv)]


def check_scores_tsv(path, label):
    header = path.read_text().splitlines()[0]
    if header != HEADER:
        fail(f"{label}: header is not the pinned beta2016 column set")
    table = rows(path)
    if len(table) != NPOSES:
        fail(f"{label}: {len(table)} rows, expected {NPOSES}")
    if [row["name"] for row in table] != NAMES:
        fail(f"{label}: rows are not the inputs, in order, named for them")
    totals = [float(row["total"]) for row in table]
    if not all(math.isfinite(total) for total in totals):
        fail(f"{label}: a total is not finite")
    spread = max(totals) - min(totals)
    if spread > POSE_SPREAD:
        fail(f"{label}: identical poses spread {spread:.6f} > {POSE_SPREAD}")
    worst = max(abs(total - REFERENCE_TOTAL) for total in totals)
    if worst > TOTAL_TOLERANCE:
        fail(f"{label}: total differs from the pinned reference by "
             f"{worst:.6f} > {TOTAL_TOLERANCE}")
    return totals


def check_verb_outdir(directory, label, lowered_by):
    scores = directory / "scores.tsv"
    if not scores.is_file():
        fail(f"{label}: {scores} is missing")
    structures = sorted(p.name for p in directory.iterdir()
                        if p.name != "scores.tsv")
    if structures != NAMES:
        fail(f"{label}: not one written structure per input, named for it")
    totals = totals_of(scores)
    if not all(math.isfinite(total) for total in totals):
        fail(f"{label}: a total is not finite")
    least = min(REFERENCE_TOTAL - total for total in totals)
    if least < lowered_by:
        fail(f"{label}: a pose was lowered by only {least:.3f} < {lowered_by}")


def pdb_coords(path):
    values = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            values.append((float(line[30:38]), float(line[38:46]),
                           float(line[46:54])))
    return values


def check_seeded_pair():
    a = {p.name: p.read_bytes() for p in (T / "pack-a").iterdir()}
    b = {p.name: p.read_bytes() for p in (T / "pack-b").iterdir()}
    if a == b:
        return  # byte-identical: the strongest outcome
    if SEEDED_MAX_TOTAL_DELTA is None:
        fail("seeded pack pair: outputs are not byte-identical")
    if set(a) != set(b):
        fail("seeded pack pair: different file sets")
    totals_a = totals_of(T / "pack-a" / "scores.tsv")
    totals_b = totals_of(T / "pack-b" / "scores.tsv")
    worst_total = max(abs(x - y) for x, y in zip(totals_a, totals_b))
    worst_coord = 0.0
    for name in NAMES:
        coords_a = pdb_coords(T / "pack-a" / name)
        coords_b = pdb_coords(T / "pack-b" / name)
        if len(coords_a) != len(coords_b):
            fail(f"seeded pack pair: {name} atom counts differ")
        worst_coord = max(
            [worst_coord] + [abs(x - y)
                             for pa, pb in zip(coords_a, coords_b)
                             for x, y in zip(pa, pb)]
        )
    if worst_total > SEEDED_MAX_TOTAL_DELTA:
        fail(f"seeded pack pair: totals differ by {worst_total:.6f} "
             f"> {SEEDED_MAX_TOTAL_DELTA}")
    if worst_coord > SEEDED_MAX_COORD_DELTA:
        fail(f"seeded pack pair: coordinates differ by "
             f"{worst_coord:.6f} > {SEEDED_MAX_COORD_DELTA} A")


def check_chain():
    written = (T / "min" / "scores.tsv").read_bytes()
    rescored = (T / "score-min.tsv").read_bytes()
    if written == rescored:
        return
    if CHAIN_EXACT:
        fail("tmol score on minimize's outdir does not reproduce "
             "scores.tsv byte for byte")
    worst = max(abs(x - y) for x, y in zip(
        totals_of(T / "min" / "scores.tsv"), totals_of(T / "score-min.tsv")))
    fail(f"chained scores differ by {worst:.6f}")


def check_demo():
    lines = (T / "demo.txt").read_text().splitlines()
    if (len(lines) != 2 or not lines[0].startswith("before ")
            or not lines[1].startswith("after ")):
        fail("demo: expected 'before <x>' and 'after <y>' lines")
    before = float(lines[0].split()[1])
    after = float(lines[1].split()[1])
    if abs(before - REFERENCE_TOTAL) > TOTAL_TOLERANCE:
        fail("demo: the single-pose score is off the batch reference")
    if REFERENCE_TOTAL - after < DEMO_LOWER_BY:
        fail(f"demo: relax lowered the score by only "
             f"{REFERENCE_TOTAL - after:.3f} < {DEMO_LOWER_BY}")


def main():
    if HEADER is None:
        raise SystemExit("check.py: reference values not pinned yet")
    totals = check_scores_tsv(T / "scores.tsv", "score")
    check_verb_outdir(T / "min", "minimize", MIN_LOWER_BY)
    check_verb_outdir(T / "pack-a", "pack", PACK_LOWER_BY)
    check_verb_outdir(T / "relax", "relax", RELAX_LOWER_BY)
    check_seeded_pair()
    check_chain()
    check_demo()
    if not (T / "demo-relaxed.pdb").is_file():
        fail("demo: the relaxed structure was not written")
    print(f"tmol test OK: 26 poses of 1UBQ at {totals[0]:.3f}; "
          "minimize/pack/relax each lowered it; the seeded pair "
          "reproduces; scores.tsv chains through tmol score")


if __name__ == "__main__":
    main()
