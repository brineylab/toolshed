#!/usr/bin/env python3
"""Generate the deterministic synthetic antibody-antigen fixture for tools/ssrmsd/test."""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import gemmi
import numpy as np

AA1_TO_AA3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

AG_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLI"  # 30 aa
HC_SEQ = "EVQLVESGGGLVQPGGSLRLSCAAS"  # 25 aa
LC_SEQ = "DIQMTQSPSSLSASVGDRVTITCRA"  # 25 aa


def helix_coords(n: int, offset: tuple[float, float, float], phase: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) * 1.7453 + phase
    x = 2.3 * np.cos(t) + offset[0]
    y = 2.3 * np.sin(t) + offset[1]
    z = 1.5 * np.arange(n, dtype=np.float64) + offset[2]
    return np.column_stack([x, y, z])


def rotation_matrix(axis: tuple[float, float, float], angle_rad: float) -> np.ndarray:
    u = np.asarray(axis, dtype=np.float64)
    u = u / np.linalg.norm(u)
    ux, uy, uz = u
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    omc = 1.0 - c
    return np.array(
        [
            [c + ux * ux * omc, ux * uy * omc - uz * s, ux * uz * omc + uy * s],
            [uy * ux * omc + uz * s, c + uy * uy * omc, uy * uz * omc - ux * s],
            [uz * ux * omc - uy * s, uz * uy * omc + ux * s, c + uz * uz * omc],
        ],
        dtype=np.float64,
    )


def make_complex(
    ab_shift_x: float,
    *,
    global_rot: np.ndarray | None = None,
    global_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    chain_ids: tuple[str, str, str] = ("A", "B", "C"),
    reverse_order: bool = False,
) -> list[tuple[str, str, np.ndarray]]:
    ag = helix_coords(len(AG_SEQ), (0.0, 0.0, 0.0), 0.0)
    hc = helix_coords(len(HC_SEQ), (15.0, 5.0, 0.0), 1.0) + np.array([ab_shift_x, 0.0, 0.0])
    lc = helix_coords(len(LC_SEQ), (15.0, -5.0, 0.0), 2.0) + np.array([ab_shift_x, 0.0, 0.0])
    gshift = np.asarray(global_shift, dtype=np.float64)
    if global_rot is not None:
        ag = ag @ global_rot.T + gshift
        hc = hc @ global_rot.T + gshift
        lc = lc @ global_rot.T + gshift
    else:
        ag = ag + gshift
        hc = hc + gshift
        lc = lc + gshift
    chains = [
        (chain_ids[0], AG_SEQ, ag),
        (chain_ids[1], HC_SEQ, hc),
        (chain_ids[2], LC_SEQ, lc),
    ]
    if reverse_order:
        chains = [chains[1], chains[2], chains[0]]
    return chains


def build_structure(chains: list[tuple[str, str, np.ndarray]]) -> gemmi.Structure:
    st = gemmi.Structure()
    st.name = "FIXT"
    model = gemmi.Model(1)
    for chain_id, seq, coords in chains:
        chain = gemmi.Chain(chain_id)
        for idx, (aa, xyz) in enumerate(zip(seq, coords, strict=True), start=1):
            res = gemmi.Residue()
            res.name = AA1_TO_AA3[aa]
            res.seqid = gemmi.SeqId(str(idx))
            res.entity_type = gemmi.EntityType.Polymer
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            atom.occ = 1.0
            atom.b_iso = 20.0
            res.add_atom(atom)
            chain.add_residue(res)
        model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    return st


def write_cif(path: Path, chains: list[tuple[str, str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_structure(chains).make_mmcif_document().as_string(), encoding="utf-8")


def write_pdb(path: Path, chains: list[tuple[str, str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_structure(chains).make_pdb_string(), encoding="utf-8")


def analytical_metrics(
    shifts_by_source: list[tuple[str, float]],
) -> dict[str, float | int | str]:
    """Exact metrics when aligning on rigid chain A and scoring translated chains B,C."""
    n = len(shifts_by_source)
    mat = np.zeros((n, n), dtype=np.float64)
    upper: list[float] = []
    within: list[float] = []
    cross: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(shifts_by_source[i][1] - shifts_by_source[j][1])
            mat[i, j] = mat[j, i] = d
            upper.append(d)
            if shifts_by_source[i][0] == shifts_by_source[j][0]:
                within.append(d)
            else:
                cross.append(d)
    u = np.asarray(upper, dtype=np.float64)
    mean_sq = float(np.mean(u**2))
    return {
        "n_structures": n,
        "n_ca_align": len(AG_SEQ),
        "n_ca_score": len(HC_SEQ) + len(LC_SEQ),
        "mean_sq_rmsd": round(mean_sq, 4),
        "rms_rmsd": round(float(np.sqrt(mean_sq)), 4),
        "mean_rmsd": round(float(np.mean(u)), 4),
        "median_rmsd": round(float(np.median(u)), 4),
        "rmsd_sd": round(float(np.std(u, ddof=0)), 4),
        "ssRMSD": round(float(np.sum(mat**2)), 4),
        "SD": round(float(np.std(mat, ddof=0)), 4),
        "within_source_mean_rmsd": round(float(np.mean(within)), 4),
        "cross_source_mean_rmsd": round(float(np.mean(cross)), 4),
    }


def main() -> None:
    fixture_dir = Path(__file__).resolve().parent / "fixture"
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)

    rot_esm = rotation_matrix((1.0, 2.0, 3.0), 1.05)
    rot_pdb = rotation_matrix((-2.0, 1.0, 0.5), 0.8)

    ensembles = {
        "mab1": ([0.0, 1.0, 2.0], [3.0, 5.0]),
        "mab2": ([0.0, 0.5, 1.5], [2.5, 4.0]),
        "mab3": ([0.0, 2.0, 4.0], [1.0, 3.0]),
    }

    expected: dict[str, dict[str, float | int | str]] = {}

    for eid, (af3_shifts, esm_shifts) in ensembles.items():
        job_dir = fixture_dir / "af3" / eid
        for idx, shift in enumerate(af3_shifts):
            write_cif(
                job_dir / f"seed-1_sample-{idx}" / f"{eid}_seed-1_sample-{idx}_model.cif",
                make_complex(shift),
            )
        # Decoy top-level duplicate with huge shift (50 A); must be auto-excluded by AF3 discovery.
        write_cif(job_dir / f"{eid}_model.cif", make_complex(50.0))

        # Flat esmfold2 layout: <eid>_sample_<i>.cif with permuted chain IDs & rigid transform.
        for idx, shift in enumerate(esm_shifts):
            write_cif(
                fixture_dir / "esmfold2" / f"{eid}_sample_{idx}.cif",
                make_complex(
                    shift,
                    global_rot=rot_esm,
                    global_shift=(12.0, -7.5, 4.2),
                    chain_ids=("X", "H", "L"),
                    reverse_order=True,
                ),
            )

        shifts_labeled = [("af3", s) for s in af3_shifts] + [("esmfold2", s) for s in esm_shifts]
        expected[eid] = analytical_metrics(shifts_labeled)

    # Sidecars produced by esmfold2 beside .cif files; discover_batch_dirs must ignore them.
    (fixture_dir / "esmfold2" / "mab1_sample_0.json").write_text('{"plddt": 0.92}\n', encoding="utf-8")
    (fixture_dir / "esmfold2" / "mab1_sample_0.npz").write_bytes(b"sidecar-placeholder")

    # Single-ensemble fixture: archived AF3 .tar.gz (with decoy) + poly-G-linked .pdb at shift 3.0.
    single_dir = fixture_dir / "single"
    single_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(single_dir / "mab1_af3.tar.gz", "w:gz") as tf:
        tf.add(fixture_dir / "af3" / "mab1", arcname="mab1")

    linked_parts = make_complex(3.0, global_rot=rot_pdb, global_shift=(-6.0, 9.0, 2.5))
    linker_seq = "G" * 15
    linker1 = helix_coords(15, (5.0, 20.0, 0.0), 0.3)
    linker2 = helix_coords(15, (10.0, -20.0, 0.0), 0.7)
    linked_seq = linked_parts[0][1] + linker_seq + linked_parts[1][1] + linker_seq + linked_parts[2][1]
    linked_xyz = np.vstack(
        [linked_parts[0][2], linker1, linked_parts[1][2], linker2, linked_parts[2][2]]
    )
    write_pdb(single_dir / "mab1_linked.pdb", [("A", linked_seq, linked_xyz)])
    expected["mab1_single"] = analytical_metrics(
        [("af3", 0.0), ("af3", 1.0), ("af3", 2.0), ("esmfold", 3.0)]
    )

    # Homomer fixture (chains B and C identical) to verify v1 loud refusal even under --align-chains A --score-chains B,C.
    hom_chains = [
        ("A", AG_SEQ, helix_coords(len(AG_SEQ), (0.0, 0.0, 0.0), 0.0)),
        ("B", HC_SEQ, helix_coords(len(HC_SEQ), (15.0, 5.0, 0.0), 1.0)),
        ("C", HC_SEQ, helix_coords(len(HC_SEQ), (15.0, -5.0, 0.0), 2.0)),
    ]
    write_cif(fixture_dir / "homomer" / "hom_0.cif", hom_chains)
    write_cif(fixture_dir / "homomer" / "hom_1.cif", hom_chains)

    (fixture_dir / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote fixture to {fixture_dir}")


if __name__ == "__main__":
    main()
