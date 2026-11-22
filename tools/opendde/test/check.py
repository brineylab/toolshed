"""check.py <out_dir>: opendde's prediction of the HIV-1 protease homodimer has the files, chains, and trained-weight confidence the skill documents."""
import json
import math
import sys
from pathlib import Path

from biotite.structure.io import pdbx

# HIV-1 protease (PDB 1HHP), a 99-residue obligate homodimer, single-sequence: 198 residues,
# past cuEquivariance's 100-residue crossover, so the default run takes the Triton kernels.
SEQ = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
PLDDT_FLOOR = 40.0  # opendde reports plddt on 0-100; set from the first real run, with margin

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

print(f"ok: hivpr: 2 chains x {len(SEQ)} residues, plddt {conf['plddt']:.1f}, "
      f"ptm {conf['ptm']:.3f}, iptm {conf['iptm']:.3f}, ranking_score {conf['ranking_score']:.3f}")
