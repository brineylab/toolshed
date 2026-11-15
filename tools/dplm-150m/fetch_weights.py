#!/usr/bin/env python3
"""Fetch this tool's pinned weights from huggingface.co into the image.

The repository, revision, and every file's sha256 and byte count are pinned
below, from the HF tree API at that revision, and each file is verified
against them after download, so the image holds exactly the pinned bytes. A
failed attempt leaves its partial file behind and the next resumes it with a
Range request. DNS, proxies, free disk, each redirect hop, and each retry
are printed, so a build that cannot fetch its weights says why in its log.

Usage: fetch_weights.py DEST [FILE ...]
With no FILE, every pinned file is fetched. A name in CONVERT_TO_BF16 ships
in fp32: it is downloaded to a staging directory beside DEST, verified, cast
to bf16 -- the rounding from_pretrained applies at load -- and saved under
DEST, with the fp32 bytes deleted before the step ends, so they never enter
an image layer.
"""
import hashlib
import os
import pathlib
import re
import shutil
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = "airkingbd/dplm_150m"
REV = "49b7125a5d28c6418fcc2f3c4fe799352ac1488b"
FILES = {  # file name -> (sha256, bytes), pinned from the HF tree API at REV
    "config.json": ("36480be9ea79156026ce374ceb9573e0edde62dfccd22d32b107e4de0fd7f03c", 792),
    "tokenizer_config.json": ("7d849f80e0f87303b55f370737e3475390c61ae55f4326db1de310711bd6c7e6", 1126),
    "special_tokens_map.json": ("3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1", 125),
    "vocab.txt": ("0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03", 93),
    "pytorch_model.bin": ("ea4eaa99536b60ed76f945f71a1a5e604f08447ec3def5104a93ca6001a59961", 595359662),
}
CONVERT_TO_BF16 = []  # names whose fp32 bytes are cast to bf16 before they enter the image

ATTEMPTS = 4          # per file
READ_TIMEOUT = 60     # seconds, per blocking socket operation
CHUNK = 1 << 20
HEARTBEAT = 30.0      # seconds between progress lines while a file streams
UA = "brineylab-toolshed-build"


def log(*parts):
    print(*parts, flush=True)


def _digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _host(url):
    return urllib.parse.urlsplit(url).netloc


def diagnostics(need_bytes):
    log(f"python {sys.version.split()[0]}")
    try:
        ips = ",".join(sorted({a[4][0] for a in socket.getaddrinfo("huggingface.co", 443, type=socket.SOCK_STREAM)}))
        log(f"dns huggingface.co -> {ips}")
    except Exception as e:
        log(f"dns huggingface.co FAILED: {type(e).__name__}: {e}")
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"):
        value = os.environ.get(var)
        if value:
            log(f"env {var}={re.sub(r'//[^/@]+@', '//***@', value)}")
    for path in ("/", "/opt"):
        try:
            usage = shutil.disk_usage(path)
            log(f"disk {path}: {usage.free / 2 ** 30:.1f} GiB free of {usage.total / 2 ** 30:.1f} GiB")
        except Exception as e:
            log(f"disk {path} FAILED: {type(e).__name__}: {e}")
    log(f"fetching {need_bytes} bytes from {REPO}@{REV[:12]}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def trace(name, url):
    """Print each hop of the redirect chain, so the log names where a fetch dies."""
    opener = urllib.request.build_opener(_NoRedirect)
    hop = url
    for _ in range(6):
        try:
            with opener.open(urllib.request.Request(hop, method="HEAD", headers={"User-Agent": UA}),
                             timeout=READ_TIMEOUT) as r:
                log(f"trace {name}: {r.status} {hop[:110]}")
                return
        except urllib.error.HTTPError as e:
            location = e.headers.get("Location") if e.code in (301, 302, 303, 307, 308) else None
            if not location:
                log(f"trace {name}: HTTP {e.code} {e.reason}")
                return
            hop = urllib.parse.urljoin(hop, location)
            log(f"trace {name}: {e.code} {_host(url)} -> {_host(hop)}")
        except Exception as e:
            log(f"trace {name}: {type(e).__name__}: {e}")
            return
    log(f"trace {name}: too many redirects")


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
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"
    want_sha, want_bytes = FILES[name]
    for attempt in range(1, ATTEMPTS + 1):
        if attempt == 3:
            _prefer_ipv4()
        try:
            if dest.exists():
                size = dest.stat().st_size
                if size == want_bytes:
                    got = _digest(dest)
                    if got == want_sha:
                        log(f"fetch {name}: complete ({size} bytes, sha256 verified)")
                        return
                    dest.unlink()
                    log(f"fetch {name}: complete file failed sha256; refetching")
                elif size > want_bytes:
                    dest.unlink()
            if not dest.exists() or dest.stat().st_size != want_bytes:
                trace(name, url)
            have = dest.stat().st_size if dest.exists() else 0
            headers = {"User-Agent": UA}
            if have:
                headers["Range"] = f"bytes={have}-"
            started = time.monotonic()
            done = 0
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=READ_TIMEOUT) as r:
                resume = have and r.status == 206
                if have and not resume:
                    have = 0  # the server ignored the range: start over
                expected = have + int(r.headers.get("Content-Length") or 0)
                if expected and expected != want_bytes:
                    raise RuntimeError(f"server offers {expected} bytes, pinned {want_bytes}")
                log(f"fetch {name}: {r.status} {('resuming from ' + str(have) + ' of ') if resume else ''}"
                    f"{want_bytes} bytes from {_host(r.url)}")
                last = started
                with open(dest, "ab" if resume else "wb") as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - last > HEARTBEAT:
                            rate = (have + done) / max(now - started, 1e-9) / 2 ** 20
                            log(f"fetch {name}: {(have + done) / 2 ** 20:.0f} MiB at {rate:.1f} MiB/s")
                            last = now
            elapsed = max(time.monotonic() - started, 1e-9)
            size = dest.stat().st_size
            if size != want_bytes:
                raise RuntimeError(f"{size} bytes on disk, pinned {want_bytes}")
            got = _digest(dest)
            if got != want_sha:
                raise RuntimeError(f"sha256 {got}, pinned {want_sha}")
            log(f"fetch {name}: {size} bytes in {elapsed:.1f}s ({size / elapsed / 2 ** 20:.1f} MiB/s), "
                f"sha256 verified")
            return
        except Exception as e:
            log(f"fetch {name}: attempt {attempt}/{ATTEMPTS} failed: {type(e).__name__}: {e}")
            if attempt == ATTEMPTS:
                raise
            time.sleep(min(2 ** attempt, 20))


def to_bf16(src, dest):
    import torch

    state = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in state.items()}, dest)
    return dest.stat().st_size


def main(argv):
    dest = pathlib.Path(argv[1])
    names = argv[2:] or list(FILES)
    unknown = [n for n in names if n not in FILES]
    if unknown:
        raise SystemExit(f"unknown file(s): {unknown}")
    diagnostics(sum(FILES[n][1] for n in names))
    dest.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / (dest.name + "-fp32-staging")
    for n in names:
        if n in CONVERT_TO_BF16:
            staging.mkdir(parents=True, exist_ok=True)
            fp32 = staging / n
            download(n, fp32)
            bf16 = dest / n
            size = to_bf16(fp32, bf16)
            log(f"cast {n}: fp32 {fp32.stat().st_size} bytes -> bf16 {size} bytes")
            fp32.unlink()
            if not any(staging.iterdir()):
                shutil.rmtree(staging)
        else:
            download(n, dest / n)
    log(f"OK: {len(names)} file(s), {sum((dest / n).stat().st_size for n in names)} bytes under {dest}")


if __name__ == "__main__":
    main(sys.argv)
