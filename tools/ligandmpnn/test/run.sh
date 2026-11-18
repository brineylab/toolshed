#!/usr/bin/env bash
# LigandMPNN's test. The fixture is 1BC8: the SAP-1 ETS domain (chain C, 93
# residues, author numbering 1-93) bound to two DNA chains (A, B) with two
# zinc ions — a real multi-chain ligand context, not a bare protein. Three
# runs exercise every documented command: a plain design run, a fixed-residue
# run with side-chain packing and stats, and a multi-PDB run. check.py then
# verifies the outputs, including that the fixed residues stay fixed.
set -euo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d /tmp/ligandmpnn.XXXXXX)
trap 'rm -rf "$T"' EXIT

ligandmpnn --model_type ligand_mpnn --pdb_path 1BC8.pdb --out_folder "$T/basic" \
  --temperature 0.2 --seed 0 --batch_size 16 --number_of_batches 2

ligandmpnn --model_type ligand_mpnn --pdb_path 1BC8.pdb --out_folder "$T/fixed" \
  --fixed_residues "C1 C2 C3 C4 C5" --chains_to_design C \
  --temperature 0.2 --seed 0 --batch_size 8 --number_of_batches 1 \
  --save_stats 1 --pack_side_chains 1

printf '{"%s/1BC8.pdb": ""}\n' "$PWD" > "$T/pdbs.json"
ligandmpnn --model_type ligand_mpnn --pdb_path_multi "$T/pdbs.json" \
  --out_folder "$T/multi" --temperature 0.2 --seed 0 --batch_size 4 \
  --number_of_batches 1

python3 check.py "$T"
