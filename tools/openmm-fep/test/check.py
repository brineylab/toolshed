"""Checks for the openmm-fep test (run as `python3 -I check.py run <tmpdir>` or `python3 -I check.py mapping <complex.cif>`).

`run` reads run.sh's outputs, <tmpdir>/{bound,unbound,k44a,g38a,pdb}/<leg>/replicate-0. The fixture's chain A is MET1-LEU53
with LEU47, LYS44 and GLY38, and its glycan (one GlcNAc, 0YB, on NLN A17); chain B carries the core on B13
(4YB x2, VMB, 0MA x2), the other five glycan residues. `mapping` builds the L47P mapping with no cycles.
"""

import importlib.machinery
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

from openmm.app import PDBxFile

BOUND_GLYCANS = Counter({"0YB": 1, "4YB": 2, "VMB": 1, "0MA": 2})
UNBOUND_GLYCANS = Counter({"0YB": 1})
GLYCAM = set(BOUND_GLYCANS)
ALA = {"N", "CA", "C", "O", "CB", "H", "HA", "HB1", "HB2", "HB3"}


def residues(path):
    """{(chain, id): (name, [atom names])} of an end-state mmCIF."""
    top = PDBxFile(str(path)).topology
    return {(r.chain.id, r.id): (r.name, [a.name for a in r.atoms()]) for r in top.residues()}


def check_alanine(leg_dir, chain, resid, wt_name, extra=0):
    """Mutant residue is ALA with exactly ALA's atoms; every other residue has identical atom names; the count drops by the side chain's difference (returned)."""
    wt, mut = residues(leg_dir / "wildtype.cif"), residues(leg_dir / "mutant.cif")
    assert wt[(chain, resid)][0] == wt_name, (leg_dir, wt[(chain, resid)][0])
    assert mut[(chain, resid)][0] == "ALA", (leg_dir, mut[(chain, resid)][0])
    names = mut[(chain, resid)][1]
    assert len(names) == len(set(names)) and set(names) == ALA, (leg_dir, sorted(names))
    assert set(wt) == set(mut), leg_dir
    for key in wt:
        if key != (chain, resid):
            assert wt[key] == mut[key], (leg_dir, key)
    sizes = [sum(len(v[1]) for v in end.values()) for end in (wt, mut)]
    assert sizes[0] - sizes[1] == len(wt[(chain, resid)][1]) - len(ALA), (leg_dir, sizes)
    return sizes[0] - sizes[1]


def check_glycans(leg_dir, expected, chains):
    for end in ("wildtype", "mutant"):
        res = residues(leg_dir / f"{end}.cif")
        found = Counter(v[0] for v in res.values() if v[0] in GLYCAM)
        assert found == expected, (leg_dir, end, found)
        assert {c for c, _ in res} == chains, (leg_dir, end, {c for c, _ in res})


def check_result(leg_dir):
    from gufe.protocols import ProtocolDAGResult
    from gufe.tokenization import JSON_HANDLER
    from feflow.protocols import NonEquilibriumCyclingProtocol
    with open(leg_dir / "result.json") as handle:
        result = ProtocolDAGResult.from_dict(json.load(handle, cls=JSON_HANDLER.decoder))
    assert result.ok(), leg_dir
    data = NonEquilibriumCyclingProtocol(NonEquilibriumCyclingProtocol.default_settings()).gather([result]).data
    for side in ("forward_work", "reverse_work"):
        works = [float(w[-1]) for w in data[side]]
        assert works and all(math.isfinite(w) for w in works), (leg_dir, side, works)
    run = json.loads((leg_dir / "run.json").read_text())
    assert set(run) == {"mutation", "leg", "partner", "cycles", "replicate", "input_sha256", "versions", "settings"}, run
    assert run["settings"]["alchemical_settings"]["explicit_charge_correction"] is True
    assert run["settings"]["num_cycles"] == run["cycles"] == 1


def check_log_names_ion(leg_dir, ion):
    """K44A removes a +1 charge, so feflow's charge difference (wild type minus mutant) is +1: a Na+ is needed."""
    log = (leg_dir / "log.txt").read_text()
    assert "alchemical water" in log and f"a {ion} ion" in log, log[-2000:]


def check_pdb_matches_cif(pdb_dir, cif_dir):
    for end in ("wildtype", "mutant"):
        a, b = residues(pdb_dir / f"{end}.cif"), residues(cif_dir / f"{end}.cif")
        assert sorted((c, v[0], len(v[1])) for (c, _), v in a.items()) == sorted((c, v[0], len(v[1])) for (c, _), v in b.items()), end


def check_mapping(fixture):
    """L47P: H is unmapped (proline has none); N, CA, C, O, CB, HA are mapped; every other atom is."""
    path = shutil.which("openmm-fep")
    loader = importlib.machinery.SourceFileLoader("openmm_fep", path)
    wrapper = importlib.util.module_from_spec(importlib.util.spec_from_loader("openmm_fep", loader))
    loader.exec_module(wrapper)
    from gufe import ProteinComponent
    top, pos = wrapper.load(fixture)
    target = wrapper.check_target(top, "A", "L", "47", "")
    mut_top, mut_pos = wrapper.mutate(top, pos, target, "PRO")
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        wrapper.write_cif(Path(tmp) / "wt.cif", top, pos)
        wrapper.write_cif(Path(tmp) / "mut.cif", mut_top, mut_pos)
        wt = ProteinComponent.from_pdbx_file(str(Path(tmp) / "wt.cif")).to_rdkit()
        mut = ProteinComponent.from_pdbx_file(str(Path(tmp) / "mut.cif")).to_rdkit()
    mapping = wrapper.atom_mapping(wt, mut, ("A", "47"))
    def in_target(atom):
        info = atom.GetPDBResidueInfo()
        return info.GetChainId() == "A" and info.GetResidueNumber() == 47
    mapped = {a.GetPDBResidueInfo().GetName().strip() for a in wt.GetAtoms() if in_target(a) and a.GetIdx() in mapping}
    assert mapped == {"N", "CA", "C", "O", "CB", "HA"}, mapped
    assert "H" in {a.GetPDBResidueInfo().GetName().strip() for a in wt.GetAtoms() if in_target(a)}
    outside = [a for a in wt.GetAtoms() if not in_target(a)]
    assert all(a.GetIdx() in mapping for a in outside)
    assert len(mapping) == len(outside) + 6
    print("mapping ok")


def main():
    if sys.argv[1] == "mapping":
        return check_mapping(sys.argv[2])
    t = Path(sys.argv[2])
    bound, unbound = t / "bound/bound/replicate-0", t / "unbound/unbound/replicate-0"
    k44a, g38a, pdb = t / "k44a/bound/replicate-0", t / "g38a/bound/replicate-0", t / "pdb/unbound/replicate-0"
    assert check_alanine(bound, "A", "47", "LEU") == 9
    assert check_alanine(unbound, "A", "47", "LEU") == 9
    check_alanine(k44a, "A", "44", "LYS")
    check_alanine(g38a, "A", "38", "GLY")
    check_glycans(bound, BOUND_GLYCANS, {"A", "B", "C", "D", "E", "F", "G", "H"})
    check_glycans(unbound, UNBOUND_GLYCANS, {"A", "C"})
    for leg in (bound, unbound, k44a, g38a, pdb):
        check_result(leg)
    check_log_names_ion(k44a, "Na+")
    check_pdb_matches_cif(pdb, unbound)
    print("run ok")


if __name__ == "__main__":
    main()
