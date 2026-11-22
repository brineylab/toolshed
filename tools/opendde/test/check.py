"""check.py <out_dir>: opendde's predictions of the HIV-1 protease homodimer, single-sequence and with the fixture's
MSA, have the files, chains, folded geometry, and trained-weight confidence the skill documents, and the MSA run
lands on the crystal structure."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from biotite.structure.io import pdb, pdbx

# HIV-1 protease (PDB 1HHP), a 99-residue obligate homodimer: 198 residues, past cuEquivariance's 100-residue
# crossover, so the default run takes the Triton kernels.
SEQ = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
PLDDT_FLOOR = 30.0  # 0-100; single-sequence hivpr scored 42.0-42.7 over 5 samples on an L40S
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
    A trunk that miscomputes cannot fold it right, which chain geometry and confidence alone do not catch."""
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


def check(out, job, against_crystal):
    cifs = sorted(out.glob(f"**/{job}_sample_*.cif"))
    confs = sorted(out.glob(f"**/{job}_summary_confidence_sample_*.json"))
    assert len(cifs) == 1, f"{job}: expected 1 structure, found {[str(p) for p in cifs]}"
    assert len(confs) == 1, f"{job}: expected 1 summary confidence, found {[str(p) for p in confs]}"

    conf = json.loads(confs[0].read_text())
    for k in ("plddt", "ptm", "iptm", "ranking_score", "has_clash"):
        assert k in conf, f"{confs[0].name} lacks {k}: keys {sorted(conf)}"
        assert math.isfinite(float(conf[k])), (job, k, conf[k])
    for k in ("ptm", "iptm"):
        assert 0.0 <= conf[k] <= 1.0, (job, k, conf[k])
    assert 0.0 <= conf["plddt"] <= 100.0, (job, conf["plddt"])
    assert conf["plddt"] > PLDDT_FLOOR, f"{job}: plddt {conf['plddt']:.1f} is low for a known homodimer: the weights look untrained"

    atoms = pdbx.get_structure(pdbx.CIFFile.read(str(cifs[0])), model=1)
    ca = atoms[atoms.atom_name == "CA"]
    chains = sorted(set(ca.chain_id))
    assert len(chains) == 2, f"{job}: chains {chains}, expected two"
    assert len(ca) == 2 * len(SEQ), f"{job}: {len(ca)} CA atoms, expected {2 * len(SEQ)}"
    assert_folded(job, ca)
    line = (f"ok: {job}: 2 chains x {len(SEQ)} residues, plddt {conf['plddt']:.1f}, ptm {conf['ptm']:.3f}, "
            f"iptm {conf['iptm']:.3f}, ranking_score {conf['ranking_score']:.3f}")
    if against_crystal:
        rmsd = crystal_rmsd(ca)
        assert rmsd < RMSD_LIMIT, f"{job}: CA RMSD {rmsd:.1f} A to the crystal (PDB 1HVR): the fold is wrong"
        line += f", CA RMSD {rmsd:.2f} A to the crystal"
    print(line)


out = Path(sys.argv[1])
check(out, "hivpr", against_crystal=False)  # single-sequence folds it wrong (see crystal_rmsd)
check(out, "hivpr_msa", against_crystal=True)
