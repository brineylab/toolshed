#!/usr/bin/env python3
"""Fetch this tool's pinned weights from huggingface.co into the image.

The repository, revision, and every file's sha256 and byte count are pinned
below, from the HF tree API at that revision, and each file is verified
against them after download, so the image holds exactly the pinned bytes. A
failed attempt leaves its partial file behind and the next resumes it with a
Range request. DNS, proxies, free disk, each redirect hop, and each retry
are printed, so a build that cannot fetch its weights says why in its log.

Usage: fetch_weights.py DEST [FILE ...]
With no FILE, every pinned file is fetched, verified against its pin, and
left exactly as released: the wrapper casts to bf16 at load, the same cast
the test's reference map was computed with.
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

REPO = "airkingbd/dplm_650m"
REV = "7a7e651baa667d094aba05e9dc1cf52a3332110a"
FILES = {  # file name -> (sha256, bytes), pinned from the HF tree API at REV
    "config.json": ("ecc4ec5f5d741d99d0653da16ca95a188c4065941393ed6e3d69336ca33db557", 793),
    "tokenizer_config.json": ("7d849f80e0f87303b55f370737e3475390c61ae55f4326db1de310711bd6c7e6", 1126),
    "special_tokens_map.json": ("3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1", 125),
    "vocab.txt": ("0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03", 93),
    "pytorch_model.bin": ("db4e54343a89e7600f41c3aacbc593db1b0caee82ec28cab25ff2ae090eba39c", 2609617722),
}
# The checkpoint is stored exactly as released (fp32); the wrapper casts it to
# bf16 at load with torch_dtype=bfloat16, as the test's reference did.

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


def disk_report():
    for path in ("/", "/opt"):
        try:
            usage = shutil.disk_usage(path)
            log(f"disk {path}: {usage.free / 2 ** 30:.1f} GiB free of {usage.total / 2 ** 30:.1f} GiB")
        except Exception as e:
            log(f"disk {path} FAILED: {type(e).__name__}: {e}")


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
    disk_report()
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


def main(argv):
    dest = pathlib.Path(argv[1])
    names = argv[2:] or list(FILES)
    unknown = [n for n in names if n not in FILES]
    if unknown:
        raise SystemExit(f"unknown file(s): {unknown}")
    diagnostics(sum(FILES[n][1] for n in names))
    dest.mkdir(parents=True, exist_ok=True)
    for n in names:
        download(n, dest / n)  # the pinned bytes, verified against the pin
    log(f"OK: {len(names)} file(s), {sum((dest / n).stat().st_size for n in names)} bytes under {dest}")
    log("disk after everything landed:")
    disk_report()


if __name__ == "__main__":
    main(sys.argv)
