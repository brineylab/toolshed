#!/usr/bin/env python3
"""Fetch and verify AbLang1's pinned heavy or light weight archive."""
import hashlib
import pathlib
import sys
import tarfile
import urllib.request

PINS = {
    "heavy": (318636334, "7c939d690cadbd2464901f80f58bf7018a6cd2b7edd73ff2ba6eb29f97ffa4f7"),
    "light": (318291943, "b5310bc75903feb781ef88adf3003f0f65595755755afa56b81c41a957c50cc9"),
}


def main():
    chain, destination = sys.argv[1:]
    destination = pathlib.Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"ablang-{chain}.tar.gz"
    url = f"https://opig.stats.ox.ac.uk/data/downloads/ablang-{chain}.tar.gz"
    with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    expected_bytes, expected_sha256 = PINS[chain]
    assert archive.stat().st_size == expected_bytes, (archive.stat().st_size, expected_bytes)
    assert digest.hexdigest() == expected_sha256, (digest.hexdigest(), expected_sha256)
    # The verified archive contains exactly the three files ablang.pretrained reads.
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        assert {member.name for member in members} == {"amodel.pt", "hparams.json", "vocab.json"}
        assert all(member.isfile() and "/" not in member.name for member in members)
        package.extractall(destination, filter="data")
    archive.unlink()
    model = destination / "amodel.pt"
    print(f"AbLang1 {chain} model sha256:", hashlib.sha256(model.read_bytes()).hexdigest())
    print(f"AbLang1 {chain} files:", sorted(path.name for path in destination.iterdir()))


if __name__ == "__main__":
    main()
