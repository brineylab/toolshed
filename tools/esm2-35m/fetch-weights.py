#!/usr/bin/env python3
"""Fetch esm2-35m's pinned weights from huggingface.co into the image.

The repository, revision, and the weights file's sha256 and byte count are pinned
below, from the Hub tree API at that revision (the LFS oid the Hub itself serves),
and the file is verified against its pin after download, so the image holds exactly
the pinned bytes. A failed attempt leaves its partial file behind and the next
resumes it with a Range request. Each retry prints its error, so a fetch that cannot
succeed says why. The small files (config, tokenizer, vocab) are committed beside
this script and copied in after the fetch, so changing them never invalidates the
weights layer.

Usage: fetch-weights.py DEST [FILE ...]
"""
import hashlib
import pathlib
import socket
import sys
import time
import urllib.request

REPO = "facebook/esm2_t12_35M_UR50D"
REV = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"
FILES = {  # file name -> (sha256, bytes), pinned from the Hub tree API at REV
    "model.safetensors": ("e35647818e0e064351d4531ed480d225a002567b4b2b93ad3a9246d753150fc0", 136008798),
}
# The weights are stored exactly as released (fp32); the wrapper loads them in bf16
# for contacts, the same rounding oplm's own computation runs on.

HOSTS = ["huggingface.co", "hf-mirror.com"]  # tried in turn; both serve /resolve ranges
ATTEMPTS = 4          # per file
READ_TIMEOUT = 60     # seconds, per blocking socket operation
CHUNK = 1 << 20
UA = "brineylab-toolshed-build"


def log(*parts):
    print(*parts, flush=True)


def _digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _prefer_ipv4():
    """Retry with IPv4 addresses only: some networks route IPv6 into the ground."""
    if getattr(socket.getaddrinfo, "_ipv4_only", False):
        return
    original = socket.getaddrinfo

    def ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return original(host, port, socket.AF_INET, type, proto, flags)

    ipv4_only._ipv4_only = True
    socket.getaddrinfo = ipv4_only
    log("note: further attempts resolve IPv4 addresses only")


def download(name, dest):
    want_sha, want_bytes = FILES[name]
    for attempt in range(1, ATTEMPTS + 1):
        if attempt == 3:
            _prefer_ipv4()
        host = HOSTS[(attempt - 1) % len(HOSTS)]
        url = f"https://{host}/{REPO}/resolve/{REV}/{name}"
        try:
            if dest.exists():
                size = dest.stat().st_size
                if size == want_bytes:
                    got = _digest(dest)
                    if got == want_sha:
                        return
                    dest.unlink()
                    log(f"fetch {name}: complete file failed sha256; refetching")
                elif size > want_bytes:
                    dest.unlink()
            have = dest.stat().st_size if dest.exists() else 0
            headers = {"User-Agent": UA}
            if have:
                headers["Range"] = f"bytes={have}-"
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=READ_TIMEOUT) as r:
                resume = have and r.status == 206
                if have and not resume:
                    have = 0  # the server ignored the range: start over
                expected = have + int(r.headers.get("Content-Length") or 0)
                if expected and expected != want_bytes:
                    raise RuntimeError(f"server offers {expected} bytes, pinned {want_bytes}")
                with open(dest, "ab" if resume else "wb") as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
            size = dest.stat().st_size
            if size != want_bytes:
                raise RuntimeError(f"{size} bytes on disk, pinned {want_bytes}")
            got = _digest(dest)
            if got != want_sha:
                raise RuntimeError(f"sha256 {got}, pinned {want_sha}")
            return
        except Exception as e:
            log(f"fetch {name}: attempt {attempt}/{ATTEMPTS} ({host}) failed: "
                f"{type(e).__name__}: {e}")
            if attempt == ATTEMPTS:
                raise
            time.sleep(min(2 ** attempt, 20))


def main(argv):
    dest = pathlib.Path(argv[1])
    names = argv[2:] or list(FILES)
    unknown = [n for n in names if n not in FILES]
    if unknown:
        raise SystemExit(f"unknown file(s): {unknown}")
    dest.mkdir(parents=True, exist_ok=True)
    for n in names:
        download(n, dest / n)  # the pinned bytes, verified against the pin


if __name__ == "__main__":
    main(sys.argv)
