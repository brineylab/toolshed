"""Fixture facts for the openmm test; the checks themselves come in later tasks.

complex.cif is Boltz-2's model_0 for complex.yaml, unmodified (CPU, seed 0, single-sequence).
It has no _struct_conn loop, so no glycosidic or Asn-NAG bond is recorded: the test must find
them by distance. Boltz writes each sugar as its own chain (C-H), HETATM, residue number 1.

Asn sites, as (chain, residue number): the lone NAG (chain C) on A:17, the core on B:13.
Core, from the Asn: D NAG, E NAG, F BMA, G MAN (alpha1-3), H MAN (alpha1-6).
"""

ASN_SITES = [("A", 17), ("B", 13)]

# whole residues of both chains with a heavy atom within 4.5 A of the other chain, computed from
# complex.cif; the termini A53 and B28 are left out to keep the region at 24 residues
ML_REGION = ("A:20-21,A:24-24,A:28-28,A:31-31,A:35-35,A:40-40,A:44-44,A:46-47,A:50-51,"
             "B:4-5,B:8-8,B:11-12,B:15-16,B:19-20,B:22-23,B:26-26")

# Asp/Glu -1, Lys/Arg +1, His neutral at pH 7 (none present), counted from the residue types:
# A: E21 K28 K35 K44 E51 = +1; B: E4 E11 R22 = -1; caps are neutral
ML_REGION_CHARGE = 0

# aimnet 0.2.6 calculators/calculator.py:209 `nb_threshold: int = 120,` and :1932
# `or self.nb_threshold < N`: more than 120 atoms switches from dense to neighbor list.
# The region (24 residues, about 17 atoms each with hydrogens, about 400 atoms) is above it.
AIMNET_NB_THRESHOLD = 120
