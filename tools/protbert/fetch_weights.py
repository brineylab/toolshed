#!/usr/bin/env python3
"""Put ProtBERT's pinned fp32 checkpoint at /opt/model/pytorch_model.bin.

Seven builds of this image have died on the machine that builds it, and what
they taught is this file's design:

- Three died downloading (e01a2db, 80a06ff, fc447aa): connections to
  huggingface.co that open and then move no bytes, for the 1.7 GB checkpoint
  and the 81-byte vocab.txt alike, while PyPI, the base image and ghcr.io all
  pull normally. When the CDN does answer, it delivers at 60 MB/s.
- Four died in or just after the weights step (87ec561, 567fbb0, f51f62e,
  ac2a6ca): one at the 10 KB COPY that followed a committed 1.7 GB layer,
  two in the weights step itself, all silently, with no error a log shows.

Every one of those builds could run long: a stalled first attempt cost
minutes before a retry delivered, and the later builds tried their sources
one after another, each with a deadline of minutes, and then converted the
checkpoint to bf16 in the same step. So this build assumes the builder kills
a build that runs too long and spends as few minutes as it can:

- The three sources -- huggingface.co, its mirror hf-mirror.com, and
  ProtTrans's own Zenodo record 4633691, three different network ends -- are
  raced at once, each in its own thread. The fastest source ends the step; a
  stall costs only as much time as the slowest source needs to be overtaken,
  not the sum of every source's deadline.
- No conversion runs at all. The image ships the fp32 pytorch_model.bin, and
  the wrapper loads it with dtype=bfloat16 -- bit for bit the same bf16
  weights a build-time conversion would ship, verified on CPU in a campaign
  run that compares the cast-at-load state dict and a forward pass against
  the converted safetensors. Twice the bytes on disk, minutes off the build.
- Every read is timed: one that moves no bytes for 30 s is a stalled
  connection and is dropped; an attempt never runs longer than 3 minutes;
  a source never runs longer than 9; and whichever racer delivers bytes that
  match the pinned sha256 moves them into place and ends the race.

The tokenizer, config and license files are not fetched: the five of them are
11 KB and are committed in tools/protbert/model/ (their sha256s are checked
below), so the network moves one file, the checkpoint. The first lines this
script prints are the builder's own numbers (RAM, /dev/shm and model-dir free
space), because the head of the log is all a failed build shows.
"""
import hashlib
import os
import shutil
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

MODEL_DIR = "/opt/model"
OUT = os.path.join(MODEL_DIR, "pytorch_model.bin")

# Rostlab/prot_bert @ 7a894481acdc12202f0a415dd567f6cfdb698908
PINNED_SHA256 = "6ea3edd26cfefc3111176100ee2a027ed510f9c86b63e5b53a2a050b59f2af9d"
PINNED_SIZE = 1684058277  # the LFS oid and byte count of that revision's pytorch_model.bin

# tools/protbert/model/, as committed: the small files of the same revision,
# and the license that ships beside the weights.
SMALL_FILES_SHA256 = {
    "LICENSE": "77b837ef9c78d0bdf63dad6c2db164de950ccbf56feaacd1436bb0a3feb3630b",
    "config.json": "44c1e1229d520841e5b29f1dd2292595219061daf0c5f5dacb4f68d1272d69a3",
    "tokenizer_config.json": "73f5e97ab29042bac49dc96f7b8a7a790994509bbd2fe3e9cd4f59affc46c2ce",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "vocab.txt": "7ea18f4fe3a796c1146f9830d50334a251f368db96bdd0606e78846ca3a30607",
}

SOURCES = [
    {
        "name": "huggingface.co",
        "url": "https://huggingface.co/Rostlab/prot_bert/resolve/"
               "7a894481acdc12202f0a415dd567f6cfdb698908/pytorch_model.bin",
        "kind": "bin",
        "size": PINNED_SIZE,
        "hash": ("sha256", PINNED_SHA256),
    },
    {
        "name": "hf-mirror.com",
        "url": "https://hf-mirror.com/Rostlab/prot_bert/resolve/"
               "7a894481acdc12202f0a415dd567f6cfdb698908/pytorch_model.bin",
        "kind": "bin",
        "size": PINNED_SIZE,
        "hash": ("sha256", PINNED_SHA256),
    },
    {
        "name": "zenodo.org",
        "url": "https://zenodo.org/api/records/4633691/files/prot_bert.zip/content",
        "kind": "zip",
        "size": 1561909078,  # the record's own file listing, with its md5
        "hash": ("md5", "1bc4f9ae39e0d38bb43ecf049337893f"),
    },
]

READ_TIMEOUT_S = 30    # a read that moves no bytes for half a minute is a stalled connection
ATTEMPT_CAP_S = 180    # however slowly an attempt moves, it never runs longer than this
SOURCE_DEADLINE_S = 540  # a racer gives up once this much of the step has gone by
CHUNK = 1 << 20
SHM_MIN_FREE = 4 << 30  # stage in /dev/shm only with this much spare there (a roomy builder)

# Set once some racer has delivered the pinned bytes into OUT.
won = threading.Event()


def log(message):
    print("[fetch_weights %s] %s" % (time.strftime("%H:%M:%S"), message), file=sys.stderr, flush=True)


def file_digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def free_bytes(path):
    """Spare bytes on path's filesystem; 0 when it cannot be read."""
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except OSError:
        return 0


def ram_total_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def gb(n):
    return "%.1f GB" % (n / 1e9)


def download(src, path):
    """Stream src's url to path until it holds src's bytes, resuming across
    attempts. True when the byte count is right; the caller checks the hash,
    which is the real completion criterion."""
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have == src["size"]:
        return True
    if have > src["size"]:
        log("%s: partial file is %d bytes, past the expected %d; starting over"
            % (src["name"], have, src["size"]))
        os.remove(path)
        have = 0
    headers = {"User-Agent": "brineylab-toolshed-protbert-build"}
    if have:
        headers["Range"] = "bytes=%d-" % have
    req = urllib.request.Request(src["url"], headers=headers)
    started = time.monotonic()
    noted = started
    got = 0
    with urllib.request.urlopen(req, timeout=READ_TIMEOUT_S) as r:
        host = urllib.parse.urlsplit(r.url).netloc
        if have and r.status != 206:
            log("%s: resume at %d bytes got HTTP %s, not 206; starting over" % (src["name"], have, r.status))
            os.remove(path)
            return False
        if not have and r.status not in (200, 206):
            raise urllib.error.HTTPError(r.url, r.status, "unexpected status", r.headers, None)
        log("%s: HTTP %s from %s, content-length %s"
            % (src["name"], r.status, host, r.headers.get("Content-Length")))
        with open(path, "ab" if have else "wb") as f:
            while not won.is_set():
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                now = time.monotonic()
                if now - started > ATTEMPT_CAP_S:
                    log("%s: %d of %d bytes after %d s; dropping this attempt"
                        % (src["name"], have + got, src["size"], ATTEMPT_CAP_S))
                    return False
                if now - noted >= 15:
                    log("%s: %d of %d bytes, %.1f MB/s"
                        % (src["name"], have + got, src["size"], got / (now - started) / 1e6))
                    noted = now
    now = os.path.getsize(path)
    if now == src["size"]:
        log("%s: the file holds %d of %d bytes" % (src["name"], now, src["size"]))
        return True
    log("%s: %d of %d bytes; short read, retrying with resume" % (src["name"], now, src["size"]))
    return False


def extract_bin(zip_path, out_path):
    """Pull pytorch_model.bin out of ProtTrans's prot_bert.zip into out_path."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [i for i in zf.infolist() if i.filename.endswith("pytorch_model.bin")]
        if not members:
            log("the zip holds no pytorch_model.bin: %s" % zf.namelist())
            return None
        member = max(members, key=lambda i: i.file_size)
        log("zenodo.org: extracting %s (%d bytes) from the zip" % (member.filename, member.file_size))
        if member.file_size != PINNED_SIZE:
            log("the zip's pytorch_model.bin is %d bytes, not the pinned %d" % (member.file_size, PINNED_SIZE))
            return None
        with zf.open(member) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)
    return out_path


def deliver(candidate):
    """Move a racer's pinned checkpoint into place and end the race."""
    if file_digest(candidate, "sha256") != PINNED_SHA256:
        log("the delivered checkpoint does not match the pinned sha256; these bytes are not shipped")
        os.remove(candidate)
        return False
    try:
        os.replace(candidate, OUT)
    except OSError:  # staging sits on another filesystem (a tmpfs)
        shutil.copyfile(candidate, OUT)
        os.remove(candidate)
    if not won.is_set():
        won.set()
        log("the pinned checkpoint is in place at %s (%d bytes)" % (OUT, os.path.getsize(OUT)))
    return True


def racer(src, stage):
    """Fetch from one source until it delivers the pinned bytes, another
    racer wins, or the step's deadline passes."""
    target = os.path.join(stage, "from-%s.%s" % (src["name"].split(".")[0], src["kind"]))
    deadline = time.monotonic() + SOURCE_DEADLINE_S
    attempt = 0
    while not won.is_set() and time.monotonic() < deadline:
        attempt += 1
        try:
            if not download(src, target):
                time.sleep(2)
                continue
            algo, want = src["hash"]
            got = file_digest(target, algo)
            log("%s: %s of the downloaded file: %s (expected %s)" % (src["name"], algo, got, want))
            if got != want:
                log("%s: hash mismatch; these bytes are not shipped" % src["name"])
                os.remove(target)
                continue
            if src["kind"] == "zip":
                candidate = extract_bin(target, os.path.join(stage, "from-zenodo.bin"))
                os.remove(target)
                if candidate is None:
                    continue
            else:
                candidate = target
            deliver(candidate)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout,
                TimeoutError, ConnectionError, OSError) as e:
            log("%s: attempt %d failed: %s: %s" % (src["name"], attempt, type(e).__name__, e))
            time.sleep(2)


def main():
    # The builder's own numbers go first: a failed build shows only the head
    # of its log, and these are the numbers that say which resource ran out.
    ram, shm_free, fs_free = ram_total_bytes(), free_bytes("/dev/shm"), free_bytes(MODEL_DIR)
    stage = "/dev/shm/protbert-fetch" if shm_free >= SHM_MIN_FREE else os.path.join(MODEL_DIR, ".downloads")
    log("python %s; model dir %s" % (sys.version.split()[0], MODEL_DIR))
    log("builder: %s RAM; /dev/shm %s free; model-dir filesystem %s free"
        % (gb(ram), gb(shm_free), gb(fs_free)))
    log("racing %s for the fp32 checkpoint, staged at %s (%s)"
        % (", ".join(s["name"] for s in SOURCES), stage,
           "RAM, not the store's disk" if stage.startswith("/dev/shm") else "the layer filesystem"))

    for name, want in sorted(SMALL_FILES_SHA256.items()):
        path = os.path.join(MODEL_DIR, name)
        got = file_digest(path, "sha256") if os.path.exists(path) else None
        if got != want:
            log("FATAL: %s is missing or changed (sha256 %s, expected %s)" % (name, got, want))
            return 1
    log("the committed config, tokenizer and license files match the pinned revision")

    if os.path.exists(OUT):
        if file_digest(OUT, "sha256") == PINNED_SHA256:
            log("%s is already in place; nothing to do" % OUT)
            return 0
        os.remove(OUT)  # wrong bytes never become the checkpoint

    os.makedirs(stage, exist_ok=True)
    threads = [threading.Thread(target=racer, args=(src, stage), daemon=True) for src in SOURCES]
    started = time.monotonic()
    for t in threads:
        t.start()
    # A racer that has delivered spends its last seconds hashing and moving
    # the checkpoint; the others stop within one read timeout of the win.
    for t in threads:
        t.join(timeout=max(0.0, started + SOURCE_DEADLINE_S + 60 - time.monotonic()))

    if won.is_set() and os.path.exists(OUT) and file_digest(OUT, "sha256") == PINNED_SHA256:
        shutil.rmtree(stage, ignore_errors=True)
        with open(os.path.join(MODEL_DIR, ".weights_source"), "w") as f:
            f.write("Rostlab/prot_bert @ 7a894481acdc12202f0a415dd567f6cfdb698908\n"
                    "fp32 pytorch_model.bin sha256 %s\n"
                    "shipped as fp32; the wrapper casts it to bf16 at load\n" % PINNED_SHA256)
        log("weights in place at %s" % OUT)
        return 0
    log("FATAL: no source delivered the pinned checkpoint in %d s" % SOURCE_DEADLINE_S)
    return 1


if __name__ == "__main__":
    sys.exit(main())
