"""Offline checks of the bioemu2 calls the skill documents.

Every command in run.sh must have written topology.cif + samples.xtc of the
right shape, finite coordinates and a physicality report of the right length,
without any network. Output checks: the fixture's known system sizes (10-residue
chignolin; 214-residue adenylate kinase plus its AP5 ligand in the 1AKE assembly)
and shape/plumbing rather than exact coordinates, since samples are stochastic
(the test seeds them at reduced denoising steps to fit the 30-minute budget).
Also asserts the cached MSA really fed the local-A3M run.
"""
import json
import sys
from pathlib import Path

import mdtraj
import numpy as np
import yaml
from biotite.structure.io import pdbx


def frames(out: Path, expected_frames: int, expected_residues: int, expected_chains: int):
    for name in ("topology.cif", "samples.xtc", "physicality.json", "system.yaml", "command.txt"):
        assert (out / name).is_file(), f"missing {out / name}"
    trajectory = mdtraj.load(str(out / "samples.xtc"), top=str(out / "topology.cif"))
    assert trajectory.n_frames == expected_frames, (out, trajectory.n_frames)
    assert trajectory.topology.n_residues == expected_residues, (
        out, trajectory.topology.n_residues, expected_residues)
    assert trajectory.topology.n_chains == expected_chains, (out, trajectory.topology.n_chains)
    assert np.isfinite(trajectory.xyz).all(), f"{out}: non-finite coordinates"
    assert float(trajectory.xyz.std()) > 1e-6, f"{out}: all atoms at one point"
    report = json.loads((out / "physicality.json").read_text())
    reports = list(report.values()) if isinstance(report, dict) else report
    if reports and isinstance(reports[0], dict):
        statuses = [entry.get("status") for entry in reports]
    else:
        statuses = reports
    assert len(statuses) == expected_frames, (out, statuses)
    return trajectory


def main() -> None:
    root = Path(sys.argv[1])
    structure = frames(root / "predict", 1, 10, 1)
    # BioEmu2 writes all heavy atoms but no hydrogens: 76 for this sequence.
    assert structure.n_atoms == 76, structure
    system = yaml.safe_load((root / "predict/system.yaml").read_text())
    assert system["entities"][0]["sequence"] == "GYDPETGTWG", system
    # The local single-sequence A3M the wrapper injected, in the saved command.
    assert "/chignolin.a3m" in (root / "predict/command.txt").read_text(), "no --msa used"
    ensemble = frames(root / "ensemble", 2, 10, 1)
    assert ensemble.n_atoms == structure.n_atoms
    endpoints = frames(root / "propagate", 2, 10, 1)
    assert endpoints.n_atoms == structure.n_atoms
    assert np.allclose(endpoints.time, 1000.0), endpoints.time[:4]
    # MDTraj prefers RCSB's author chain IDs; 1AKE gives proteins A and B
    # but ligand C the same author ID B, so its topology reloads as two
    # chains. The output CIF must retain the system's canonical A,B,C.
    direct = frames(root / "pdb", 1, 215, 2)
    assert direct.n_atoms > 3000, direct
    assert {r.name for r in direct.topology.residues} >= {"AP5"}, "1AKE's AP5 ligand missing"
    pdb_system = yaml.safe_load((root / "pdb/system.yaml").read_text())
    entities = pdb_system["entities"]
    assert sorted(e["asym_ids"] for e in entities) == [["A", "B"], ["C"]], entities
    assert entities[1].get("ccd_code") == "AP5", entities
    cif = pdbx.CIFFile.read(str(root / "pdb/topology.cif"))
    atom_site = cif[next(iter(cif))]["atom_site"]
    assert set(atom_site["label_asym_id"].as_array(str)) == {"A", "B", "C"}
    print(f"predict-structure: 1 chignolin frame, {structure.n_atoms} atoms")
    print(f"sample-equilibrium: 2 chignolin frames, {ensemble.n_atoms} atoms")
    print(f"propagate: 2 endpoints at 1 ns, {endpoints.n_atoms} atoms")
    print(f"predict-structure --pdb-id 1AKE: homodimer chains A,B + AP5 ligand C, "
          f"215 residues, {direct.n_atoms} atoms")
    print("all checks passed")


if __name__ == "__main__":
    main()
