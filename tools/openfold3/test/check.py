"""check.py <out_dir>: openfold3's prediction of the HIV-1 protease homodimer has the files, chains, and trained-weight confidence the skill documents."""
import json
import math
import sys
from pathlib import Path

from biotite.structure.io import pdbx

# HIV-1 protease (PDB 1HHP), a 99-residue obligate homodimer, single-sequence: 198 residues,
# past cuEquivariance's 100-residue crossover, so the default run takes its kernels.
SEQ = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
PLDDT_FLOOR = 40.0  # set from the first real run, with margin

out = Path(sys.argv[1])
cifs = sorted(out.glob("hivpr/seed_*/hivpr_seed_*_sample_*_model.cif"))
aggs = sorted(out.glob("hivpr/seed_*/hivpr_seed_*_sample_*_confidences_aggregated.json"))
assert len(cifs) == 1, f"expected 1 structure, found {[str(p) for p in cifs]}"
assert len(aggs) == 1, f"expected 1 aggregated confidence, found {[str(p) for p in aggs]}"

conf = json.loads(aggs[0].read_text())
for k in ("avg_plddt", "ptm", "iptm", "sample_ranking_score", "has_clash"):
    assert k in conf, f"{aggs[0].name} lacks {k}: keys {sorted(conf)}"
    assert math.isfinite(float(conf[k])), (k, conf[k])
for k in ("ptm", "iptm"):
    assert 0.0 <= conf[k] <= 1.0, (k, conf[k])
assert 0.0 <= conf["avg_plddt"] <= 100.0, conf["avg_plddt"]
assert conf["avg_plddt"] > PLDDT_FLOOR, f"avg_plddt {conf['avg_plddt']:.1f} is low for a known homodimer: the weights look untrained"

atoms = pdbx.get_structure(pdbx.CIFFile.read(str(cifs[0])), model=1)
ca = atoms[atoms.atom_name == "CA"]
chains = sorted(set(ca.chain_id))
assert chains == ["A", "B"], f"chains {chains}, expected A and B"
assert len(ca) == 2 * len(SEQ), f"{len(ca)} CA atoms, expected {2 * len(SEQ)}"

print(f"ok: hivpr: 2 chains x {len(SEQ)} residues, avg_plddt {conf['avg_plddt']:.1f}, "
      f"ptm {conf['ptm']:.3f}, iptm {conf['iptm']:.3f}, sample_ranking_score {conf['sample_ranking_score']:.3f}")
