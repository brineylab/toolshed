"""check.py <out_dir>: esmfold2's predictions of the fixture's three jobs have the files, chains, and trained-weight confidence the skill documents."""
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
PLDDT_FLOOR = 0.4  # on esmfold2's own pLDDT scale; set from the first real run, with margin

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
    print(f"ok: {name}: {len(chains)} chain(s) x {len(seq)} residues, plddt {conf['plddt']:.3f}, "
          f"ptm {conf['ptm']:.3f}" + (f", iptm {conf['iptm']:.3f}" if len(chains) > 1 else ""))
