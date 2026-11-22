"""check.py <out_dir>: esmfold2's predictions of the fixture's three jobs have the files, chains, and folded geometry and trained-weight confidence the skill documents."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from biotite.structure.io import pdbx

# HIV-1 protease (PDB 1HHP), a 99-residue obligate homodimer: 198 residues, single-sequence. GCN4-p1, a
# 33-residue chain, folded alone: the second job of the same file, through the same loaded model.
JOBS = {
    "hivpr": (["A", "B"], "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"),
    "gcn4": (["A"], "RMKQLEDKVEELLSKNYHLENEVARLKKLVGER"),
    # The same dimer conditioned on a five-row MSA (the query and the L10I, L33F, I54V, and A71V variants): the
    # MSA encoder's path, which the full model has and ESMFold2-Fast does not.
    "hivpr_msa": (["A", "B"], "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"),
}
PLDDT_FLOOR = 0.6  # esmfold2's pLDDT is 0-1; hivpr 0.941, hivpr_msa 0.940, gcn4 0.784 on an L40S

def assert_folded(name, ca):
    """Trained weights give chains, not a cloud: neighbouring CAs 3.6-4.0 A apart (3.73-3.81 A median on the
    L40S runs, a cis-proline the odd 2.9 A), and a dimer compact (Rg 15.6-19.1 A) with its chains touching
    (closest CA pair 4.0-6.7 A). Confidence alone cannot tell low-confidence folds from broken weights."""
    steps = np.concatenate([np.linalg.norm(np.diff(ca.coord[ca.chain_id == c], axis=0), axis=1)
                            for c in sorted(set(ca.chain_id))])
    bonded = float(np.mean((steps > 3.6) & (steps < 4.0)))
    assert bonded >= 0.95, f"{name}: only {bonded:.0%} of neighbouring CAs are 3.6-4.0 A apart: not a protein chain"
    chains = sorted(set(ca.chain_id))
    if len(chains) == 2:
        rg = float(np.sqrt(((ca.coord - ca.coord.mean(0)) ** 2).sum(1).mean()))
        assert rg < 25.0, f"{name}: radius of gyration {rg:.1f} A: not a compact fold"
        a, b = (ca.coord[ca.chain_id == c] for c in chains)
        gap = float(np.linalg.norm(a[:, None] - b[None], axis=2).min())
        assert gap < 8.0, f"{name}: the chains' closest CAs are {gap:.1f} A apart: not a dimer"


out = Path(sys.argv[1])
for name, (chains, seq) in JOBS.items():
    stem = out / f"{name}_sample_0"
    conf = json.loads(stem.with_suffix(".json").read_text())
    for k in ("plddt", "ptm"):
        assert math.isfinite(conf[k]), (name, k, conf[k])
    assert 0.0 <= conf["ptm"] <= 1.0, (name, conf["ptm"])
    if len(chains) > 1:
        assert 0.0 <= conf["iptm"] <= 1.0, (name, conf["iptm"])
        assert np.shape(conf["pair_chains_iptm"]) == (2, 2), (name, conf["pair_chains_iptm"])
    assert conf["plddt"] > PLDDT_FLOOR, f"{name}: plddt {conf['plddt']:.3f} is low for a known fold: the weights look untrained"

    arrays = np.load(stem.with_suffix(".npz"))
    n = len(chains) * len(seq)
    assert arrays["plddt"].shape[-1] == n, (name, arrays["plddt"].shape, n)
    assert arrays["pae"].shape[-2:] == (n, n), (name, arrays["pae"].shape)

    atoms = pdbx.get_structure(pdbx.CIFFile.read(str(stem.with_suffix(".cif"))), model=1)
    ca = atoms[atoms.atom_name == "CA"]
    assert sorted(set(ca.chain_id)) == chains, (name, sorted(set(ca.chain_id)))
    assert len(ca) == n, f"{name}: {len(ca)} CA atoms, expected {n}"
    assert_folded(name, ca)
    print(f"ok: {name}: {len(chains)} chain(s) x {len(seq)} residues, plddt {conf['plddt']:.3f}, "
          f"ptm {conf['ptm']:.3f}" + (f", iptm {conf['iptm']:.3f}" if len(chains) > 1 else ""))
