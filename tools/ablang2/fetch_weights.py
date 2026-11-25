#!/usr/bin/env python3
"""Fetch and verify the pinned AbLang2 paired weights package into the image."""
import hashlib
import pathlib
import sys
import tarfile
import time
import urllib.request

URL = "https://zenodo.org/api/records/10185169/files/ablang2-weights.tar.gz/content"
MD5 = "6425b35e9b83750fde67a6a6d240d996"
BYTES = 166154117


def main(destination):
    destination = pathlib.Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = pathlib.Path("/tmp/ablang2-weights.tar.gz")
    # The cluster observed two interrupted reads of this archive. Resume its
    # byte range instead of restarting, and verify its published pin.
    for attempt in range(1, 5):
        have = archive.stat().st_size if archive.exists() else 0
        if have > BYTES:
            archive.unlink()
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            request = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                resume = bool(have and response.status == 206)
                if have and not resume:
                    have = 0
                with archive.open("ab" if resume else "wb") as handle:
                    while chunk := response.read(1 << 20):
                        handle.write(chunk)
            size = archive.stat().st_size
            if size == BYTES:
                break
            raise IOError(f"downloaded {size} of {BYTES} bytes")
        except (OSError, EOFError) as error:
            print(f"weights attempt {attempt}/4: {type(error).__name__}: {error}", flush=True)
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    assert archive.stat().st_size == BYTES
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    assert digest == MD5, (digest, MD5)
    # The pinned archive is known: it contains only the paired model's regular files.
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        assert all(m.isfile() and "/" not in m.name and not m.name.startswith(".") for m in members), members
        package.extractall(destination, filter="data")
    archive.unlink()
    model = destination / "model.pt"
    print("AbLang2 paired model sha256:", hashlib.sha256(model.read_bytes()).hexdigest())
    print("AbLang2 paired files:", sorted(p.name for p in destination.iterdir()))


if __name__ == "__main__":
    main(sys.argv[1])
