"""check.py <out_dir>: opendde's prediction of the HIV-1 protease homodimer has the files, chains, and folded geometry and trained-weight confidence the skill documents."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from biotite.structure.io import pdbx

# HIV-1 protease (PDB 1HHP), a 99-residue obligate homodimer, single-sequence: 198 residues,
# past cuEquivariance's 100-residue crossover, so the default run takes the Triton kernels.
SEQ = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
PLDDT_FLOOR = 30.0  # 0-100; single-sequence hivpr scored 42.0-42.7 over 5 samples on an L40S

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


out = Path(sys.argv[1])
cifs = sorted(out.glob("**/hivpr_sample_*.cif"))
confs = sorted(out.glob("**/hivpr_summary_confidence_sample_*.json"))
assert len(cifs) == 1, f"expected 1 structure, found {[str(p) for p in cifs]}"
assert len(confs) == 1, f"expected 1 summary confidence, found {[str(p) for p in confs]}"

conf = json.loads(confs[0].read_text())
for k in ("plddt", "ptm", "iptm", "ranking_score", "has_clash"):
    assert k in conf, f"{confs[0].name} lacks {k}: keys {sorted(conf)}"
    assert math.isfinite(float(conf[k])), (k, conf[k])
for k in ("ptm", "iptm"):
    assert 0.0 <= conf[k] <= 1.0, (k, conf[k])
assert 0.0 <= conf["plddt"] <= 100.0, conf["plddt"]
assert conf["plddt"] > PLDDT_FLOOR, f"plddt {conf['plddt']:.1f} is low for a known homodimer: the weights look untrained"

atoms = pdbx.get_structure(pdbx.CIFFile.read(str(cifs[0])), model=1)
ca = atoms[atoms.atom_name == "CA"]
chains = sorted(set(ca.chain_id))
assert len(chains) == 2, f"chains {chains}, expected two"
assert len(ca) == 2 * len(SEQ), f"{len(ca)} CA atoms, expected {2 * len(SEQ)}"
assert_folded("hivpr", ca)

print(f"ok: hivpr: 2 chains x {len(SEQ)} residues, plddt {conf['plddt']:.1f}, "
      f"ptm {conf['ptm']:.3f}, iptm {conf['iptm']:.3f}, ranking_score {conf['ranking_score']:.3f}")
