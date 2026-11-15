#!/usr/bin/env python3
"""Fetch ProteinBERT's pinned pretrained dump into /opt/model.

The file is 192 MB, an order of magnitude under the gigabyte-scale steps that
killed protbert's builds on the machine that builds them, so unlike protbert's
thirteen chunk steps it crosses the builder as one short RUN step -- carrying
the same machinery that kept those steps alive: two sources that race, byte
range resume across attempts, a read that moves no bytes for READ_TIMEOUT_S
counted as a stalled connection, a deadline that fails the step having owed the
builder only that much, and a sha256 the bytes must show before the step's
layer is committed. The wrapper checks the same sha256 again before it loads
the file, so the weights are pinned twice: per build here, at run time there.

The two sources hold the same dump: the GitHub LFS media URL of
Brandes-Lab/proteinbert_data_files (whose pointer pins the sha256 below) and
Zenodo record 10371965 (the authors' archive of the same snapshot).
"""
import hashlib
import os
import socket
import sys
import time
import urllib.error
import urllib.request

MODEL_DIR = os.environ.get("PROTEINBERT_DIR", "/opt/model")  # the override is for testing this script
FILENAME = "epoch_92400_sample_23500000.pkl"

# Brandes-Lab/proteinbert_data_files @ master: the released pretrained model
# (the dump ProteinBERT's README names for the full-GO pretrained model).
PINNED_SHA256 = "cffb53e237e49fa91f19a4c8bc0898dcfd75fd12eab7833d6e95b36c02ebacbf"
PINNED_SIZE = 191800918

SOURCES = [
    ("github.com", "https://media.githubusercontent.com/media/Brandes-Lab/proteinbert_data_files/master/%s" % FILENAME),
    ("zenodo.org", "https://zenodo.org/api/records/10371965/files/full_go_%s/content" % FILENAME),
]

READ_TIMEOUT_S = 20    # a read that moves no bytes for this long is a stalled connection
STEP_DEADLINE_S = 1200  # this step fails, having owed the builder only this much
BLOCK = 1 << 20


def log(message):
    print("[fetch_weights %s] %s" % (time.strftime("%H:%M:%S"), message), file=sys.stderr, flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def note_source():
    with open(os.path.join(MODEL_DIR, ".weights_source"), "w") as f:
        f.write("Brandes-Lab/proteinbert_data_files @ master, %s\n"
                "%s sha256 %s\n"
                "(the same snapshot as Zenodo record 10371965)\n"
                % (FILENAME, FILENAME, PINNED_SHA256))


def fetch_once(name, url, target):
    """One ranged attempt from one source. True when target holds the pinned file."""
    have = os.path.getsize(target) if os.path.exists(target) else 0
    if have > PINNED_SIZE:
        log("%s: %d bytes held is past the file's %d; starting over" % (name, have, PINNED_SIZE))
        os.remove(target)
        have = 0
    headers = {"User-Agent": "brineylab-toolshed-proteinbert-build",
               "Range": "bytes=%d-" % have}
    req = urllib.request.Request(url, headers=headers)
    began = time.monotonic()
    with urllib.request.urlopen(req, timeout=READ_TIMEOUT_S) as r:
        if have and r.status != 206:
            log("%s: a resume request came back HTTP %s, not 206; starting over" % (name, r.status))
            os.remove(target)
            return False
        if not have and r.status not in (200, 206):
            log("%s: the fetch came back HTTP %s, not 200" % (name, r.status))
            return False
        content_range = r.headers.get("Content-Range", "")
        if have and not content_range.startswith("bytes %d-" % have):
            log("%s: served %r, not the range asked for" % (name, content_range))
            return False
        log("%s: %d of %d bytes already held, content-length %s"
            % (name, have, PINNED_SIZE, r.headers.get("Content-Length")))
        with open(target, "ab" if have else "wb") as f:
            while True:
                block = r.read(BLOCK)
                if not block:
                    break
                f.write(block)
                have += len(block)
    if have != PINNED_SIZE:
        log("%s: %d of %d bytes; the connection ended early" % (name, have, PINNED_SIZE))
        return False
    digest = sha256_of(target)
    if digest != PINNED_SHA256:
        log("%s: the file hashes to %s, not the pinned %s; discarding and refetching from byte 0"
            % (name, digest, PINNED_SHA256))
        os.remove(target)
        return False
    log("%s: file delivered and verified in %.1f s" % (name, time.monotonic() - began))
    return True


def fetch():
    os.makedirs(MODEL_DIR, exist_ok=True)
    target = os.path.join(MODEL_DIR, FILENAME)
    if os.path.exists(target):
        size = os.path.getsize(target)
        if size == PINNED_SIZE and sha256_of(target) == PINNED_SHA256:
            log("%s already holds its pinned %d bytes; nothing to do" % (FILENAME, size))
            note_source()
            return True
        if size >= PINNED_SIZE:
            log("%s's file holds %d bytes but not the pinned ones; refetching it" % (FILENAME, size))
            os.remove(target)
        else:
            log("%s's file holds %d of %d bytes; the next attempt resumes from there"
                % (FILENAME, size, PINNED_SIZE))

    deadline = time.monotonic() + STEP_DEADLINE_S
    attempt = 0
    while time.monotonic() < deadline:
        name, url = SOURCES[attempt % len(SOURCES)]
        attempt += 1
        try:
            if fetch_once(name, url, target):
                note_source()
                return True
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout,
                TimeoutError, ConnectionError, OSError) as e:
            log("%s: attempt %d failed: %s: %s" % (name, attempt, type(e).__name__, e))
    log("FATAL: %s (%d bytes) not delivered in %d s" % (FILENAME, PINNED_SIZE, STEP_DEADLINE_S))
    return False


def main():
    log("python %s; fetching %s, %d bytes, sha256 %s"
        % (sys.version.split()[0], FILENAME, PINNED_SIZE, PINNED_SHA256))
    return 0 if fetch() else 1


if __name__ == "__main__":
    sys.exit(main())
