#!/usr/bin/env python3
"""Fetch one chunk of ProtALBERT's pinned fp32 checkpoint into /opt/model.

Eight builds of protbert, this shed's first ProtTrans tool, died on the
machine that builds it: three on huggingface.co connections that open and
then move no bytes, five more silently in or just after a weights step that
had already run for minutes, one of them at the 10 KB COPY that followed a
committed layer. The head of a failed build's log is all it shows. So this
tool crosses the builder in seven short steps from the start, one 128 MiB
byte range of the pinned file per RUN instruction, each verified against
the manifest below before its layer is committed:

- a step that needs nothing finishes in seconds, and one the network stalls
  fails within four minutes, having owed the builder only that much;
- every layer the steps commit is 128 MiB, not the whole 0.9 GB checkpoint;
- and because these RUN instructions never change between builds, whatever
  a failed build leaves in the builder's layer cache the next build reuses,
  so progress survives a build that dies -- which one long step never could.

The two sources are the Hub and its mirror; each request is a byte range, so
an attempt that already moved bytes before the connection stalled is
resumed, not restarted. The wrapper assembles the seven parts at load time
and checks the whole file's sha256 before torch sees it, so the checkpoint
is pinned twice: per chunk here, whole at run time.
"""
import hashlib
import os
import socket
import sys
import time
import urllib.error
import urllib.request

MODEL_DIR = os.environ.get("PROTALBERT_CHUNK_DIR", "/opt/model")  # the override is for testing this script
REVISION = "2384f5574ea4b85218ae7d5e21d17957105e672e"

# Rostlab/prot_albert @ REVISION: its whole fp32 checkpoint.
PINNED_SHA256 = "5f616acbd5007236453cef55029264af6da30d8f4491a4e5f63641055e53d2b6"
PINNED_SIZE = 897396780

SOURCES = [
    ("huggingface.co", "https://huggingface.co/Rostlab/prot_albert/resolve/%s/pytorch_model.bin" % REVISION),
    ("hf-mirror.com", "https://hf-mirror.com/Rostlab/prot_albert/resolve/%s/pytorch_model.bin" % REVISION),
]

# One row per RUN step: the byte range of the pinned checkpoint that step
# delivers, and the sha256 it must show before its layer is committed.
CHUNKS = [
    (0, 134217728, "acd1a86d2f89b8700b296ef4c36791405d458ad675a15647e096d2ed9f77ef76"),
    (134217728, 134217728, "9271b4888be7d05c19ffe15c943be1591318c6ae2effa50d0c97b31255510fa3"),
    (268435456, 134217728, "b4a0dd99fe8bf56b1693f0fd1888b5206127d28eca364bafb1c54c85034ca87a"),
    (402653184, 134217728, "b18fe2d00e1760d55cfebf5f51fb2897808be553870deb9bc8fa6f819fbf3d59"),
    (536870912, 134217728, "9b3ffa2e710b55fed91cf31c99763aab7bbaa63f315b8327e18b0f0895320662"),
    (671088640, 134217728, "cc176b3fb243f3d3e454d4a67e9cf9a1af3da040fb2b8db8a9d91fa8914e6702"),
    (805306368, 92090412, "ba0b3a38aa5fa24de14c61f533c63830e4942860e0b23046ce69bf94f23031c6"),
]

READ_TIMEOUT_S = 20    # a read that moves no bytes for this long is a stalled connection
STEP_DEADLINE_S = 240  # this step fails, having owed the builder only this much
BLOCK = 1 << 20


def log(message):
    print("[fetch_chunk %s] %s" % (time.strftime("%H:%M:%S"), message), file=sys.stderr, flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def note_source():
    with open(os.path.join(MODEL_DIR, ".weights_source"), "w") as f:
        f.write("Rostlab/prot_albert @ %s\n"
                "fp32 pytorch_model.bin sha256 %s\n"
                "shipped as seven verified 128 MiB chunks; the wrapper assembles them at load\n"
                % (REVISION, PINNED_SHA256))


def fetch_once(name, url, start, length, want, target):
    """One ranged attempt from one source. True when target holds the pinned chunk."""
    have = os.path.getsize(target) if os.path.exists(target) else 0
    if have > length:
        log("%s: %d bytes held is past the chunk's %d; starting over" % (name, have, length))
        os.remove(target)
        have = 0
    end = start + length - 1
    headers = {"User-Agent": "brineylab-toolshed-protalbert-build",
               "Range": "bytes=%d-%d" % (start + have, end)}
    req = urllib.request.Request(url, headers=headers)
    began = time.monotonic()
    with urllib.request.urlopen(req, timeout=READ_TIMEOUT_S) as r:
        if r.status != 206:
            log("%s: a range request came back HTTP %s, not 206" % (name, r.status))
            return False
        content_range = r.headers.get("Content-Range", "")
        if not content_range.startswith("bytes %d-" % (start + have)):
            log("%s: served %r, not the range asked for" % (name, content_range))
            return False
        log("%s: chunk bytes %d-%d, %d already held, content-length %s"
            % (name, start + have, end, have, r.headers.get("Content-Length")))
        with open(target, "ab" if have else "wb") as f:
            while True:
                block = r.read(BLOCK)
                if not block:
                    break
                f.write(block)
                have += len(block)
    got = os.path.getsize(target)
    if got != length:
        log("%s: %d of %d bytes; the connection ended early" % (name, got, length))
        return False
    digest = sha256_of(target)
    if digest != want:
        log("%s: the chunk hashes to %s, not the pinned %s; refetching it from byte 0" % (name, digest, want))
        os.remove(target)
        return False
    log("%s: chunk delivered and verified in %.1f s" % (name, time.monotonic() - began))
    return True


def fetch(index):
    start, length, want = CHUNKS[index]
    target = os.path.join(MODEL_DIR, "pytorch_model.bin.part%02d" % index)

    if os.path.exists(target):
        size = os.path.getsize(target)
        if size == length and sha256_of(target) == want:
            log("chunk %d already holds its pinned %d bytes; nothing to do" % (index, length))
            return True
        if size >= length:
            log("chunk %d's file holds all %d of its bytes but not the pinned ones; refetching it"
                % (index, size))
            os.remove(target)
        else:
            log("chunk %d's file holds %d of %d bytes; the next attempt resumes from there"
                % (index, size, length))

    os.makedirs(MODEL_DIR, exist_ok=True)
    deadline = time.monotonic() + STEP_DEADLINE_S
    attempt = 0
    while time.monotonic() < deadline:
        name, url = SOURCES[attempt % len(SOURCES)]
        attempt += 1
        try:
            if fetch_once(name, url, start, length, want, target):
                if index == len(CHUNKS) - 1:
                    note_source()
                return True
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout,
                TimeoutError, ConnectionError, OSError) as e:
            log("%s: attempt %d failed: %s: %s" % (name, attempt, type(e).__name__, e))
    log("FATAL: chunk %d (%d bytes at offset %d) not delivered in %d s"
        % (index, length, start, STEP_DEADLINE_S))
    return False


def main():
    if len(sys.argv) != 2:
        log("usage: fetch_chunk.py <index 0-%d>" % (len(CHUNKS) - 1))
        return 2
    try:
        index = int(sys.argv[1])
    except ValueError:
        log("the chunk index is not a number: %r" % sys.argv[1])
        return 2
    if not 0 <= index < len(CHUNKS):
        log("chunk index %s is outside 0-%d" % (sys.argv[1], len(CHUNKS) - 1))
        return 2
    total = sum(row[1] for row in CHUNKS)
    if total != PINNED_SIZE:
        log("FATAL: the manifest's chunks hold %d bytes, not the pinned %d" % (total, PINNED_SIZE))
        return 1
    log("python %s; chunk %d of %d: %d bytes at offset %d"
        % (sys.version.split()[0], index, len(CHUNKS), CHUNKS[index][1], CHUNKS[index][0]))
    return 0 if fetch(index) else 1


if __name__ == "__main__":
    sys.exit(main())
