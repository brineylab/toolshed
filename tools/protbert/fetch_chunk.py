#!/usr/bin/env python3
"""Fetch one chunk of ProtBERT's pinned fp32 checkpoint into /opt/model.

Eight builds of this image died on the machine that builds it: three on
huggingface.co connections that open and then move no bytes, five more
silently in or just after a weights step that had already run for minutes,
one of them at the 10 KB COPY that followed a committed 1.7 GB layer. The
head of a failed build's log is all it shows, so the last words of every one
of those builds were a download in progress. So the checkpoint crosses the
builder in thirteen short steps, one 128 MiB byte range of the pinned file
per RUN instruction, each verified against the manifest below before its
layer is committed:

- a step that needs nothing finishes in seconds, and one the network stalls
  fails within four minutes, having owed the builder only that much;
- every layer the steps commit is 128 MiB, not 1.7 GB, so no step's diff is
  large enough to be the thing the builder dies on;
- and because these RUN instructions never change between builds, whatever
  a failed build leaves in the builder's layer cache the next build reuses,
  so progress survives a build that dies -- which one long step never could.

The two sources are the Hub and its mirror; each request is a byte range, so
an attempt that already moved bytes before the connection stalled is
resumed, not restarted. The wrapper assembles the thirteen parts at load
time and checks the whole file's sha256 before torch sees it, so the
checkpoint is pinned twice: per chunk here, whole at run time.
"""
import hashlib
import os
import socket
import sys
import time
import urllib.error
import urllib.request

MODEL_DIR = os.environ.get("PROTBERT_CHUNK_DIR", "/opt/model")  # the override is for testing this script
REVISION = "7a894481acdc12202f0a415dd567f6cfdb698908"

# Rostlab/prot_bert @ REVISION, the UniRef100 model: its whole fp32 checkpoint.
PINNED_SHA256 = "6ea3edd26cfefc3111176100ee2a027ed510f9c86b63e5b53a2a050b59f2af9d"
PINNED_SIZE = 1684058277

SOURCES = [
    ("huggingface.co", "https://huggingface.co/Rostlab/prot_bert/resolve/%s/pytorch_model.bin" % REVISION),
    ("hf-mirror.com", "https://hf-mirror.com/Rostlab/prot_bert/resolve/%s/pytorch_model.bin" % REVISION),
]

# One row per RUN step: the byte range of the pinned checkpoint that step
# delivers, and the sha256 it must show before its layer is committed.
CHUNKS = [
    (0, 134217728, "48fa410b2ac5bd1bda70cb03b58b7ddc7754e209570367470140abe04dcceac7"),
    (134217728, 134217728, "8acf425fb4c375498bb7a1cbb6e08dc1c9004c3776929117aa6029e973593c33"),
    (268435456, 134217728, "23d42760f1d93f470b353de5825e01c237faa5496920934c5d753e9a705c4e03"),
    (402653184, 134217728, "4992182d3b89fe8f0d0e75c3d5a1c7350c07fdc695f16c493232379afd1e7eda"),
    (536870912, 134217728, "7c209d200159c2cbdf414ad0e4e61124a358b89e70a8cee9be71fb00d9553e43"),
    (671088640, 134217728, "a84330963492d7114a6d9808092fa1a8ca92781a825e916fb6b8a2e198471045"),
    (805306368, 134217728, "ba954c14ca9de9420fa540266c4d0c35db36286adfab412c508804ae4162d417"),
    (939524096, 134217728, "19f579b131e60e3c186d5bbc050791d3d5371dafe4d8668c2ecd73d9f5b487d7"),
    (1073741824, 134217728, "9155bc03af068e22d21885d910ac6e406b9915ffe4e8657768956664f031bd25"),
    (1207959552, 134217728, "96492a7cf830078d37a2a6f3efc3c37c9b7a4d5833ee0bb9151cd4f5a12f97ea"),
    (1342177280, 134217728, "6517eebf81d5dbe6288a7b36cc2669e7e85414a01184f5608667c7bb6411d2b7"),
    (1476395008, 134217728, "a39b83787e0387ed8b6aa44307bccc3621d0895ffcab59578e8f7da90e95905a"),
    (1610612736, 73445541, "1c10560182ccdf430a98167cdba9eb0d53620c38b91ce9657e52a37a4bfeea43"),
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
        f.write("Rostlab/prot_bert @ %s\n"
                "fp32 pytorch_model.bin sha256 %s\n"
                "shipped as thirteen verified 128 MiB chunks; the wrapper assembles them at load\n"
                % (REVISION, PINNED_SHA256))


def fetch_once(name, url, start, length, want, target):
    """One ranged attempt from one source. True when target holds the pinned chunk."""
    have = os.path.getsize(target) if os.path.exists(target) else 0
    if have > length:
        log("%s: %d bytes held is past the chunk's %d; starting over" % (name, have, length))
        os.remove(target)
        have = 0
    end = start + length - 1
    headers = {"User-Agent": "brineylab-toolshed-protbert-build",
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
