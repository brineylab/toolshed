#!/usr/bin/env python3
"""Check tmol's test outputs against the known fixture.

The fixture is water-stripped 1UBQ (76 residues; the record's 58 waters
removed) in 26 copies. 26 crosses tmol's first input-growth threshold:
the packer chunks a pose stack whose poses hold at most 256 blocks into
chunks of 25 poses, so `tmol pack` on this batch runs the chunked path --
25 poses, then 1. Fewer than 25 poses would never leave one chunk.

The reference values are pinned by the campaign's measurement run on
this same fixture (NVIDIA L40S, driver 580.178.04; torch 2.14.1+cu126,
tmol 1.0.0 with tmol-ops-cu12 1.0.0 -- the image's pins), which also
measured what each tolerance has to cover:

- score: the 26 identical poses spread 0.000274 within one batch, and
  two runs in separate processes differ by at most 0.000183 -- GPU
  score reductions are deterministic only to that noise, never byte
  for byte, so the checks below compare within it and
  TOTAL_TOLERANCE keeps 100x headroom for a different GPU;
- minimize lowered every pose by at least 437.3, pack by 288.2, relax
  by 473.0, and the demo script by 473.9;
- two `tmol pack --seed 1` runs wrote byte-identical structures, their
  scores.tsv totals 0.000161 apart;
- `tmol score` on minimize's outdir reproduced its scores.tsv within
  0.000184 -- the same reduction noise, so the chain check tolerates
  it instead of demanding byte equality.
"""
import csv
import math
import sys
from pathlib import Path

T = Path(sys.argv[1])

NPOSES = 26
NAMES = [f"pose_{i:02d}.pdb" for i in range(1, NPOSES + 1)]

# Pinned by the measurement run described above.
HEADER = (  # a score TSV's exact header: name, total, 25 weighted terms
    "name\ttotal\tfa_ljatr\tfa_ljrep\tfa_lk\tfa_elec\thbond\t"
    "cart_lengths\tcart_angles\tcart_torsions\tcart_impropers\t"
    "cart_hxltorsions\tdisulfide\trama\tomega\tdunbrack_rot\t"
    "dunbrack_rotdev\tdunbrack_semirot\tlk_ball_iso\tlk_ball\t"
    "lk_bridge\tlk_bridge_uncpl\tref\tgen_torsions\tna_torsion\t"
    "na_torsion_well\tmetal_coordination"
)
REFERENCE_TOTAL = 195.064756  # the batch's mean total; poses spread around it
TOTAL_TOLERANCE = 0.05  # cross-run noise is 0.0004; headroom for another GPU
POSE_SPREAD = 0.01  # identical poses in one batch spread 0.000274
MIN_LOWER_BY = 400  # measured: every pose lowered by at least 437.3
PACK_LOWER_BY = 250  # measured: at least 288.2
RELAX_LOWER_BY = 400  # measured: at least 473.0
DEMO_LOWER_BY = 400  # measured: 473.9
SEEDED_MAX_TOTAL_DELTA = 0.01  # the seeded pair measured 0.000161
SEEDED_MAX_COORD_DELTA = 0.05  # the pair measured 0.0; a flipped rotamer is > 0.5
CHAIN_MAX_TOTAL_DELTA = 0.01  # the chain measured 0.000184, not byte-identical


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
        return  # byte-identical: the strongest outcome
    # Not byte-identical on GPU: score reductions carry ~1e-4 of noise
    # between processes. The written and rescored tables must still hold
    # the same totals, pose by pose.
    worst = max(abs(x - y) for x, y in zip(
        totals_of(T / "min" / "scores.tsv"), totals_of(T / "score-min.tsv")))
    if worst > CHAIN_MAX_TOTAL_DELTA:
        fail("tmol score on minimize's outdir does not reproduce scores.tsv "
             f"within {CHAIN_MAX_TOTAL_DELTA}: {worst:.6f}")


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
