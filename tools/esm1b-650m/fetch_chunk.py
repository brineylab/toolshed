#!/usr/bin/env python3
"""Fetch one chunk of ESM-1b's pinned fp32 checkpoint into /opt/model.

Builds of these images die on the machine that builds them in or after any step
that moves about a GiB or more: connections to huggingface.co that open and then
move no bytes, and steps that die silently minutes in. So the checkpoint crosses
the builder in twenty short steps, one 128 MiB byte range of the pinned file per
RUN instruction, each verified against the manifest below before its layer is
committed (the pattern protbert 1.0.3 proved):

- a step that needs nothing finishes in seconds, and one the network stalls
  fails within four minutes, having owed the builder only that much;
- every layer the steps commit is 128 MiB, not 2.6 GB, so no step's diff is
  large enough to be the thing the builder dies on;
- and because these RUN instructions never change between builds, whatever
  a failed build leaves in the builder's layer cache the next build reuses,
  so progress survives a build that dies -- which one long step never could.

The two sources are the Hub and its mirror; each request is a byte range, so
an attempt that already moved bytes before the connection stalled is resumed,
not restarted. The wrapper assembles the twenty parts at load time and checks
the whole file's sha256 before torch sees it, so the checkpoint is pinned
twice: per chunk here, whole at run time.
"""
import hashlib
import os
import socket
import sys
import time
import urllib.error
import urllib.request

MODEL_DIR = os.environ.get("ESM1B_CHUNK_DIR", "/opt/model")  # the override is for testing this script
REVISION = "7b37824baec4d3658e1df7479222a7c79b465b76"

# facebook/esm1b_t33_650M_UR50S @ REVISION, the ESM-1b model: its whole fp32 checkpoint.
PINNED_SHA256 = "ebb8a35768179581f91fc01b93cf78facba2821c3bab1452abb38ef2a7f6e7b1"
PINNED_SIZE = 2609617413

SOURCES = [
    ("huggingface.co", "https://huggingface.co/facebook/esm1b_t33_650M_UR50S/resolve/%s/pytorch_model.bin" % REVISION),
    ("hf-mirror.com", "https://hf-mirror.com/facebook/esm1b_t33_650M_UR50S/resolve/%s/pytorch_model.bin" % REVISION),
]

# One row per RUN step: the byte range of the pinned checkpoint that step
# delivers, and the sha256 it must show before its layer is committed.
CHUNKS = [
    (0, 134217728, "246a6956975183a0659ebb297f244edf382bc2ada4183772235500c1205da579"),
    (134217728, 134217728, "cd24f1a137690c768ffa94fcd69d811a6d35c19082145b0fb21c207caa9bdd30"),
    (268435456, 134217728, "68eedd5dc4a6142790e90259db025102788e10367d9c06cf1e0f96fbbb2ec55d"),
    (402653184, 134217728, "03be0982e722d21a6ca6733d81d0c4aa92b4eb38006f6055fb969a487661c4f0"),
    (536870912, 134217728, "f2de558a93aa4bd39e70365361317512f71aff9c5120be40c346501889de0657"),
    (671088640, 134217728, "719cd3e656fe592b890c0e59a868e227def457f1c7e2cc47c2c294029ed9df61"),
    (805306368, 134217728, "e47f7dceababcc87c4ae5d978ce9a5782dec47e5f136d7191b20e212dde1f3ba"),
    (939524096, 134217728, "17adac5f0984d79ef7c0051bca78794fc386b1664377ba8ba24a546ef14635dd"),
    (1073741824, 134217728, "17339cd702dc2dccd816c9d44ac0457bf122fc4b98a2edfd9e3f79bd92fc9e0a"),
    (1207959552, 134217728, "b2cbd937cddf3ec76a782c895175c0b7af88bf39bfdda0fdfdbb35581633b8fb"),
    (1342177280, 134217728, "5e90be719fdb18d3f5b8a478ffdd99e104479dcf2a1cd77fb4b6a4a066329fa8"),
    (1476395008, 134217728, "3fb76338996d823220be3e5ba113e238531a25775b9a248d27c21290fca5d7f4"),
    (1610612736, 134217728, "94c6c336aab3c69d4024e6ebac8458fcd4d5e707abea6347e1d55a2fa369eae6"),
    (1744830464, 134217728, "3c7428fe2456d224069047439ae0923bc7151cbce76a2dd0517f111e660bf7f8"),
    (1879048192, 134217728, "0348dbd3c43ff60086e1790b7dec6216c32834be8cd5bf05543cfcf981f9c955"),
    (2013265920, 134217728, "065b83305da0d5acf5cdd068a07240406319ac8cc477fc9e040b40417ce74935"),
    (2147483648, 134217728, "2168d74d25e9a9acbed53a432654b25a08c431456356b60c9aba7a40ed408baf"),
    (2281701376, 134217728, "3594b98dbd8fa01bd52a41b32f26160c2c0382ddec759ca625430f351f0d9979"),
    (2415919104, 134217728, "b7ef25bd02c7b68b868b473913bc9658fa1bb831f436f02bec0c0240697b9a35"),
    (2550136832, 59480581, "c6d246d80a954f83d9e47466054fd2c1612afa1b3eb89b42623f31eb80423895"),
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
        f.write("facebook/esm1b_t33_650M_UR50S @ %s\n"
                "fp32 pytorch_model.bin sha256 %s\n"
                "shipped as twenty verified 128 MiB chunks; the wrapper assembles them at load\n"
                % (REVISION, PINNED_SHA256))


def fetch_once(name, url, start, length, want, target):
    """One ranged attempt from one source. True when target holds the pinned chunk."""
    have = os.path.getsize(target) if os.path.exists(target) else 0
    if have > length:
        log("%s: %d bytes held is past the chunk's %d; starting over" % (name, have, length))
        os.remove(target)
        have = 0
    end = start + length - 1
    headers = {"User-Agent": "brineylab-toolshed-esm1b-build",
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
