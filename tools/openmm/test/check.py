"""Fixture facts for the openmm test; the checks themselves come in later tasks.

complex.cif is Boltz-2's model_0 for complex.yaml, unmodified (CPU, seed 0, single-sequence).
It has no _struct_conn loop, so no glycosidic or Asn-NAG bond is recorded: the test must find
them by distance. Boltz writes each sugar as its own chain (C-H), HETATM, residue number 1.

Asn sites, as (chain, residue number): the lone NAG (chain C) on A:17, the core on B:13.
Core, from the Asn: D NAG, E NAG, F BMA, G MAN (alpha1-3), H MAN (alpha1-6).
"""

ASN_SITES = [("A", 17), ("B", 13)]

# whole residues from both chains at the interface (chain A residues 20-27, FEEALKLA;
# chain B residues 4-11, ELKRLAEE); Task 6 pins the atom count after preparation
ML_REGION = "A:20-27,B:4-11"

# Asp/Glu -1, Lys/Arg +1, His neutral at pH 7 (none present), counted from the residue types:
# A: E21 E22 K25 = -1; B: E4 K6 R7 E10 E11 = -1; caps are neutral
ML_REGION_CHARGE = -2

# aimnet 0.2.6 calculators/calculator.py:209 `nb_threshold: int = 120,` and :1932
# `or self.nb_threshold < N`: more than 120 atoms switches from dense to neighbor list.
# The region (16 residues, about 15 atoms each with hydrogens) is above it.
AIMNET_NB_THRESHOLD = 120
