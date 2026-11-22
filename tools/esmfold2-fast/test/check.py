"""check.py <out_dir>: esmfold2's predictions of the fixture's two jobs have the files, chains, and folded geometry and trained-weight confidence the skill documents, and each HIV-1 protease dimer lands on the crystal structure."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from biotite.structure.io import pdb, pdbx

# HIV-1 protease (PDB 1HHP), a 99-residue obligate homodimer: 198 residues, single-sequence. GCN4-p1, a
# 33-residue chain, folded alone: the second job of the same file, through the same loaded model.
JOBS = {
    "hivpr": (["A", "B"], "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"),
    # Named with a dot, which the files must keep whole: gcn4.p1_sample_0.cif, not gcn4.cif.
    "gcn4.p1": (["A"], "RMKQLEDKVEELLSKNYHLENEVARLKKLVGER"),
}
PLDDT_FLOOR = 0.6  # esmfold2's pLDDT is 0-1; hivpr scored 0.941 and gcn4 0.777 on an L40S

CRYSTAL = Path(__file__).with_name("1hvr_ca.pdb")
RMSD_LIMIT = 3.0  # A; see crystal_rmsd


def assert_folded(name, ca):
    """Trained weights give chains, not a cloud: neighbouring CAs 3.6-4.0 A apart, and a dimer compact with its
    chains touching. opendde's and openfold3's single-sequence hivpr on an L40S (5 samples each): CA steps
    3.73-3.81 A median (a cis-proline the odd 2.9 A), Rg 15.6-19.1 A, closest A-B CA pair 4.0-6.7 A; scrambled,
    stretched, and pulled-apart (16 A and up) copies fail. Confidence alone cannot tell a low-confidence fold
    from broken weights."""
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
        assert gap < 12.0, f"{name}: the chains' closest CAs are {gap:.1f} A apart: not a dimer"


def crystal_rmsd(ca):
    """CA RMSD (A) of a predicted HIV-1 protease dimer to the crystal, PDB 1HVR (1.8 A, inhibitor-bound; its
    modified residue 67 left out), superposed, over either pairing of chains. With ColabFold's 5,397-row MSA on an
    L40S, opendde scored 0.38-0.55 A and openfold3 0.45-1.7 A over 5 samples each (1HHP, the apo crystal of this
    sequence, is 1.0 A from 1HVR per chain); single-sequence, both gave wrong folds at 15-20 A with pLDDT near 40.
    ESMFold2's language model stands in for the MSA, so every hivpr job here is held to the crystal. A trunk that
    miscomputes cannot fold it right, which chain geometry and confidence alone do not catch."""
    ref = pdb.PDBFile.read(str(CRYSTAL)).get_structure(model=1)
    keys = [(c, int(r)) for c, r in zip(ref.chain_id, ref.res_id)]
    q = ref.coord - ref.coord.mean(0)
    chains = sorted(set(ca.chain_id))
    first = int(ca.res_id.min())
    best = math.inf
    for order in (chains, chains[::-1]):
        named = dict(zip(order, ("A", "B")))
        at = {(named[c], int(r) - first + 1): x for c, r, x in zip(ca.chain_id, ca.res_id, ca.coord)}
        p = np.array([at[k] for k in keys])
        p = p - p.mean(0)
        u, _, vt = np.linalg.svd(p.T @ q)
        rot = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
        best = min(best, float(np.sqrt(((p @ rot - q) ** 2).sum(1).mean())))
    return best


out = Path(sys.argv[1])
for name, (chains, seq) in JOBS.items():
    stem = f"{name}_sample_0"
    conf = json.loads((out / f"{stem}.json").read_text())
    for k in ("plddt", "ptm"):
        assert math.isfinite(conf[k]), (name, k, conf[k])
    assert 0.0 <= conf["ptm"] <= 1.0, (name, conf["ptm"])
    if len(chains) > 1:
        assert 0.0 <= conf["iptm"] <= 1.0, (name, conf["iptm"])
        assert np.shape(conf["pair_chains_iptm"]) == (2, 2), (name, conf["pair_chains_iptm"])
    assert conf["plddt"] > PLDDT_FLOOR, f"{name}: plddt {conf['plddt']:.3f} is low for a known fold: the weights look untrained"

    arrays = np.load(out / f"{stem}.npz")
    n = len(chains) * len(seq)
    assert arrays["plddt"].shape[-1] == n, (name, arrays["plddt"].shape, n)
    assert arrays["pae"].shape[-2:] == (n, n), (name, arrays["pae"].shape)

    atoms = pdbx.get_structure(pdbx.CIFFile.read(str(out / f"{stem}.cif")), model=1)
    ca = atoms[atoms.atom_name == "CA"]
    assert sorted(set(ca.chain_id)) == chains, (name, sorted(set(ca.chain_id)))
    assert len(ca) == n, f"{name}: {len(ca)} CA atoms, expected {n}"
    assert_folded(name, ca)
    line = (f"ok: {name}: {len(chains)} chain(s) x {len(seq)} residues, plddt {conf['plddt']:.3f}, "
            f"ptm {conf['ptm']:.3f}" + (f", iptm {conf['iptm']:.3f}" if len(chains) > 1 else ""))
    if name.startswith("hivpr"):
        rmsd = crystal_rmsd(ca)
        assert rmsd < RMSD_LIMIT, f"{name}: CA RMSD {rmsd:.1f} A to the crystal (PDB 1HVR): the fold is wrong"
        line += f", CA RMSD {rmsd:.2f} A to the crystal"
    print(line)
