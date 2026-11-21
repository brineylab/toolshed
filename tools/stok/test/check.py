"""check.py <tokens_dir>: the fixture's known answer, plus the decode roundtrip.

expected.json beside this script is the known answer, built with the fixture:
per-row sequence, token count, the exact null-token and null-coordinate
positions, the carried columns, the short row's rejection, and the tokenizer
identity (preset, checkpoint and codebook sha256, codebook size, stok
version).  This checks the shard layout through stok's own read_source,
every field of expected.json against the Parquet actually written, and that
the tokens mean the structure: each accepted row's tokens are decoded with
the pretrained decoder and its codebook, superimposed (Kabsch) onto the
coordinates the tokenizer itself wrote, and must land within 1.5 A RMSD.
"""
import json
import os
import sys
from pathlib import Path

RMSD_BOUND = 1.5


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: check.py TOKENS_DIR")
    tokens = Path(sys.argv[1])
    here = Path(__file__).parent
    expected = json.loads((here / "expected.json").read_text())

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from Bio.SVDSuperimposer import SVDSuperimposer

    from stok.data.format import read_source
    from stok.models.decoder import load_pretrained_decoder
    from stok.utils.codebook import load_codebook
    from stok.utils.decoding import decode_structure_tokens
    from stok.utils.pretrained import resolve_gcp_artifact

    source = read_source(tokens)
    assert source["rows"] == 2, f"{source['rows']} rows"
    assert [s["rows"] for s in source["shards"]] == [2, 0], source["shards"]
    assert source["structure"] and source["coordinates"], "shards lack tokens or coordinates"
    assert source["columns"] == [
        "sequence_id", "sequence", "structure_tokens", "coordinates",
        "mean_plddt", "uniprot_accession",
    ], source["columns"]
    for field, value in expected["tokenizer"].items():
        actual = source["tokenizer"].get(field)
        assert actual == value, f"tokenizer {field} is {actual!r}, expected {value!r}"

    rejected = [
        json.loads(line)
        for line in (tokens / "rejected" / "00001.jsonl").read_text().splitlines()
    ]
    assert rejected == expected["rejected"], f"rejected rows {rejected}"

    table = pq.read_table(tokens / "00000.parquet").to_pylist()
    assert [row["sequence_id"] for row in table] == [
        row["sequence_id"] for row in expected["rows"]
    ], "accepted rows out of order"

    checkpoint = resolve_gcp_artifact("large", progress=False)
    codebook = load_codebook(path=str(checkpoint))
    decoder = load_pretrained_decoder("large", progress=False)
    identity = source["tokenizer"]

    record = {"identity": identity, "rejected": rejected, "rows": []}
    for row, answer in zip(table, expected["rows"]):
        sequence_id = row["sequence_id"]
        tokens_row = row["structure_tokens"]
        assert row["sequence"] == expected["source"]["sequence"], sequence_id
        assert len(tokens_row) == answer["tokens"], sequence_id
        nulls = [i for i, token in enumerate(tokens_row) if token is None]
        assert nulls == answer["null_token_positions"], (sequence_id, nulls)
        assert all(0 <= token < identity["codebook_size"]
                   for token in tokens_row if token is not None), sequence_id
        assert (row["mean_plddt"] == answer["mean_plddt"]
                and row["uniprot_accession"] == answer["uniprot_accession"]), sequence_id
        coordinates = row["coordinates"]
        assert len(coordinates) == answer["tokens"], sequence_id
        coordinate_nulls = [
            i for i, residue in enumerate(coordinates)
            if all(atom == [None, None, None] for atom in residue)
        ]
        assert coordinate_nulls == answer["coordinate_null_positions"], \
            (sequence_id, coordinate_nulls)
        for i, residue in enumerate(coordinates):
            if i not in coordinate_nulls:
                assert all(atom is not None and len(atom) == 3
                           for atom in residue), (sequence_id, i)

        indices = torch.tensor(
            [token if token is not None else -1 for token in tokens_row],
            dtype=torch.long,
        )
        decoded = decode_structure_tokens(
            decoder, codebook, indices.unsqueeze(0),
            residue_mask=torch.ones((1, len(indices)), dtype=torch.bool),
        )[0].numpy()
        truth = np.full((len(coordinates), 3, 3), np.nan, dtype=np.float32)
        for i, residue in enumerate(coordinates):
            for j, atom in enumerate(residue):
                if atom is not None:
                    truth[i, j] = atom
        keep = (np.isfinite(decoded).all(axis=(1, 2))
                & np.isfinite(truth).all(axis=(1, 2)))
        # A token exists exactly where the residue's whole backbone was
        # observed, and the coordinates column keeps N/CA/C beyond that, so
        # the superimposed set is the complete residues.
        assert keep.sum() == answer["tokens"] - len(nulls), (sequence_id, keep.sum())
        superimposer = SVDSuperimposer()
        superimposer.set(decoded[keep].reshape(-1, 3), truth[keep].reshape(-1, 3))
        superimposer.run()
        rmsd = float(superimposer.get_rms())
        assert rmsd < RMSD_BOUND, f"{sequence_id}: decode RMSD {rmsd:.3f} A"
        record["rows"].append({
            "sequence_id": sequence_id,
            "tokens": answer["tokens"],
            "null_token_positions": nulls,
            "coordinate_null_positions": coordinate_nulls,
            "decode_rmsd": round(rmsd, 3),
        })
        print(f"ok: {sequence_id}: {answer['tokens']} tokens, "
              f"{len(nulls)} null, {len(coordinate_nulls)} coordinate-null, "
              f"decode RMSD {rmsd:.3f} A")

    print(f"ok: identity preset {identity['preset']}, codebook "
          f"{identity['codebook_size']}, stok {identity['stok_version']}, "
          f"checkpoint {identity['checkpoint_sha256'][:12]}..., "
          f"codebook {identity['codebook_sha256'][:12]}...")
    print(f"ok: rejected {rejected[0]['sequence_id']} ({rejected[0]['reason']})")
    out = os.environ.get("FLYLOFT_OUT")
    if out and Path(out).is_dir():
        (Path(out) / "stok-test-record.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
