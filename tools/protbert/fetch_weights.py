#!/usr/bin/env python3
"""Put ProtBERT's weights in /opt/model as one bf16 safetensors file, pinned by sha256.

Six builds of this image have failed on the machine that builds it, and what
they taught is this file's design:

- Three died downloading (e01a2db, 80a06ff, fc447aa): connections to
  huggingface.co that open and then move no bytes, for the 1.7 GB checkpoint
  and the 81-byte vocab.txt alike, while PyPI, the base image and ghcr.io all
  pull normally. When the CDN does answer, it delivers at 60 MB/s.
- Three died in or just after the step that writes the checkpoint into the
  layer filesystem (87ec561, 567fbb0, f51f62e): one at a 10 KB COPY that
  followed the committed 1.7 GB layer, one in the weights step with nothing
  after it, all with logs unreadable past the head a failed build shows.

So this build spends as little as it can of the two resources that may be
running out. The layer filesystem never holds the fp32 checkpoint and the
safetensors at once: the checkpoint is staged in /dev/shm when that has room
(RAM, not the builder's store), and otherwise deleted the moment it is loaded,
so the store's peak is one checkpoint and its committed layer is the 0.84 GB
bf16 the computation wants. The network gets three independent sources --
huggingface.co, its mirror hf-mirror.com, and ProtTrans's own Zenodo record
4633691 -- each sha256- or md5-gated, so whichever delivers, the bytes ship
only when they are the pinned ones. And the first lines this script prints are
the builder's own numbers (RAM, /dev/shm and model-dir free space), because
the head of the log is all a failed build shows.

The fp32 original is converted in the same step that fetches it, to the same
bf16 a `dtype=torch.bfloat16` load of it gives, and deleted. If the conversion
cannot run at all, the fp32 checkpoint ships instead and the wrapper casts it
at load time: the same weights, twice the bytes, and the build still succeeds.

The tokenizer, config and license files are not fetched: the five of them are
11 KB and are committed in tools/protbert/model/ (their sha256s are checked
below), so the network moves one file, the checkpoint.
"""
import hashlib
import os
import shutil
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

MODEL_DIR = "/opt/model"
OUT = os.path.join(MODEL_DIR, "pytorch_model.bin")
SAFETENSORS = os.path.join(MODEL_DIR, "model.safetensors")

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
        "name": "huggingface.co, Rostlab/prot_bert @ 7a894481",
        "url": "https://huggingface.co/Rostlab/prot_bert/resolve/"
               "7a894481acdc12202f0a415dd567f6cfdb698908/pytorch_model.bin",
        "kind": "bin",
        "size": PINNED_SIZE,
        "hash": ("sha256", PINNED_SHA256),
        "deadline_s": 8 * 60,
    },
    {
        "name": "hf-mirror.com, the same repository and revision",
        "url": "https://hf-mirror.com/Rostlab/prot_bert/resolve/"
               "7a894481acdc12202f0a415dd567f6cfdb698908/pytorch_model.bin",
        "kind": "bin",
        "size": PINNED_SIZE,
        "hash": ("sha256", PINNED_SHA256),
        "deadline_s": 8 * 60,
    },
    {
        "name": "zenodo.org record 4633691, ProtTrans's prot_bert.zip",
        "url": "https://zenodo.org/api/records/4633691/files/prot_bert.zip/content",
        "kind": "zip",
        "size": 1561909078,  # the record's own file listing, with its md5
        "hash": ("md5", "1bc4f9ae39e0d38bb43ecf049337893f"),
        "deadline_s": 20 * 60,
    },
]

READ_TIMEOUT_S = 60   # a read that moves no bytes for a minute is a stalled connection
ATTEMPT_CAP_S = 600   # however slowly an attempt moves, it never runs longer than this
CHUNK = 1 << 20
SHM_MIN_FREE = 4 << 30  # stage in /dev/shm only with this much spare there (a roomy builder)


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


def download(url, path, expected_size, deadline_s):
    """Stream url to path until it holds expected_size bytes, resuming across
    attempts until the deadline; True when the size is right (the caller
    checks the hash, which is the real completion criterion)."""
    deadline = time.monotonic() + deadline_s
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == expected_size:
            return True
        if have > expected_size:
            log("partial file is %d bytes, past the expected %d; starting over" % (have, expected_size))
            os.remove(path)
            have = 0
        headers = {"User-Agent": "brineylab-toolshed-protbert-build"}
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            req = urllib.request.Request(url, headers=headers)
            started = time.monotonic()
            noted = started
            got = 0
            with urllib.request.urlopen(req, timeout=READ_TIMEOUT_S) as r:
                host = urllib.parse.urlsplit(r.url).netloc
                if have and r.status != 206:
                    log("resume at %d bytes got HTTP %s, not 206; starting over" % (have, r.status))
                    os.remove(path)
                    continue
                if not have and r.status not in (200, 206):
                    raise urllib.error.HTTPError(r.url, r.status, "unexpected status", r.headers, None)
                log("attempt %d: HTTP %s from %s, %d bytes present, content-length %s"
                    % (attempt, r.status, host, have, r.headers.get("Content-Length")))
                with open(path, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        now = time.monotonic()
                        if now - started > ATTEMPT_CAP_S:
                            log("attempt %d: %d of %d bytes after %d s; retrying with resume"
                                % (attempt, have + got, expected_size, ATTEMPT_CAP_S))
                            break
                        if now - noted >= 15:
                            log("attempt %d: %d of %d bytes, %.1f MB/s"
                                % (attempt, have + got, expected_size, got / (now - started) / 1e6))
                            noted = now
            now = os.path.getsize(path)
            if now == expected_size:
                log("attempt %d: the file holds %d of %d bytes" % (attempt, now, expected_size))
                return True
            log("attempt %d: %d of %d bytes; short read, retrying with resume" % (attempt, now, expected_size))
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout,
                TimeoutError, ConnectionError, OSError) as e:
            log("attempt %d failed: %s: %s" % (attempt, type(e).__name__, e))
            time.sleep(min(10, max(0.0, deadline - time.monotonic())))
    return False


def extract_bin(zip_path, out_path):
    """Pull pytorch_model.bin out of ProtTrans's prot_bert.zip into out_path."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [i for i in zf.infolist() if i.filename.endswith("pytorch_model.bin")]
        if not members:
            log("the zip holds no pytorch_model.bin: %s" % zf.namelist())
            return False
        member = max(members, key=lambda i: i.file_size)
        log("extracting %s (%d bytes) from the zip" % (member.filename, member.file_size))
        if member.file_size != PINNED_SIZE:
            log("the zip's pytorch_model.bin is %d bytes, not the pinned %d" % (member.file_size, PINNED_SIZE))
            return False
        with zf.open(member) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)
    return True


def main():
    # The builder's own numbers go first: a failed build shows only the head
    # of its log, and these are the numbers that say which resource ran out.
    ram, shm_free, fs_free = ram_total_bytes(), free_bytes("/dev/shm"), free_bytes(MODEL_DIR)
    stage = "/dev/shm/protbert-fetch" if shm_free >= SHM_MIN_FREE else os.path.join(MODEL_DIR, ".downloads")
    log("python %s; model dir %s" % (sys.version.split()[0], MODEL_DIR))
    log("builder: %s RAM; /dev/shm %s free; model-dir filesystem %s free"
        % (gb(ram), gb(shm_free), gb(fs_free)))
    log("staging the checkpoint at %s (%s)"
        % (stage, "RAM, not the store's disk" if stage.startswith("/dev/shm") else "the layer filesystem"))

    for name, want in sorted(SMALL_FILES_SHA256.items()):
        path = os.path.join(MODEL_DIR, name)
        got = file_digest(path, "sha256") if os.path.exists(path) else None
        if got != want:
            log("FATAL: %s is missing or changed (sha256 %s, expected %s)" % (name, got, want))
            return 1
    log("the committed config, tokenizer and license files match the pinned revision")

    if os.path.exists(SAFETENSORS):
        log("%s is already in place; nothing to do" % SAFETENSORS)
        return 0

    os.makedirs(stage, exist_ok=True)
    bin_path = os.path.join(stage, "weights.bin")
    if os.path.exists(bin_path) and file_digest(bin_path, "sha256") != PINNED_SHA256:
        os.remove(bin_path)  # wrong bytes never become the checkpoint
    delivered = os.path.exists(bin_path)
    if not delivered:
        for src in SOURCES:
            log("source: %s" % src["name"])
            target = os.path.join(stage, "prot_bert.zip" if src["kind"] == "zip" else "weights.bin")
            if not download(src["url"], target, src["size"], src["deadline_s"]):
                log("%s could not deliver its file in %d s" % (src["name"], src["deadline_s"]))
                continue
            algo, want = src["hash"]
            got = file_digest(target, algo)
            log("%s of the downloaded file: %s (expected %s)" % (algo, got, want))
            if got != want:
                log("hash mismatch; these bytes are not shipped")
                os.remove(target)
                continue
            if src["kind"] == "zip":
                ok = extract_bin(target, bin_path)
                os.remove(target)
                if not ok:
                    continue
            else:
                bin_path = target
            if file_digest(bin_path, "sha256") != PINNED_SHA256:
                log("the delivered checkpoint does not match the pinned sha256; these bytes are not shipped")
                os.remove(bin_path)
                continue
            delivered = True
            break
        if not delivered:
            log("FATAL: no source delivered the pinned checkpoint")
            return 1
    log("the pinned fp32 checkpoint is staged at %s (%d bytes)" % (bin_path, os.path.getsize(bin_path)))

    # Convert to one bf16 safetensors: every floating tensor cast to bf16 (the
    # cast a `dtype=torch.bfloat16` load would do), every other tensor as it
    # is. The staged file is deleted the moment it is loaded, so the staging
    # medium never holds the checkpoint and the safetensors at once.
    state, tensors = None, {}
    try:
        import torch
        from safetensors.torch import save_file

        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ValueError("the checkpoint is not a state dict: %s" % type(state))
        # the tensors are in RAM now; free the staged bytes, so the staging
        # medium never holds the checkpoint and the safetensors at once
        os.remove(bin_path)
        n_float = 0
        for key in sorted(state):
            tensor = state.pop(key)  # the fp32 dict shrinks as the bf16 one grows
            if not torch.is_tensor(tensor):
                raise ValueError("%s is a %s, not a tensor" % (key, type(tensor)))
            if torch.is_floating_point(tensor):
                tensor = tensor.to(torch.bfloat16)
                n_float += 1
            # clone, so no two keys share storage (safetensors refuses that)
            # and the saved tensors hold no reference into the fp32 checkpoint
            tensors[key] = tensor.clone().contiguous()
        tmp = SAFETENSORS + ".tmp"
        save_file(tensors, tmp, metadata={"format": "pt"})
        os.replace(tmp, SAFETENSORS)
        n_tensors = len(tensors)
        del tensors, state
    except Exception as e:
        # The conversion never sinks the build: ship the checkpoint itself,
        # which the wrapper loads with the same bf16 cast -- the same weights,
        # twice the bytes. Every tensor is either still in state (fp32, its
        # original form) or already cast to bf16 in tensors; the union is the
        # checkpoint, and a dtype=bfloat16 load gives the same weights either
        # way. Only when the load itself failed is the staged file still there.
        log("the bf16 conversion failed (%s: %s); shipping the fp32 checkpoint instead"
            % (type(e).__name__, e))
        leftover = SAFETENSORS + ".tmp"
        if os.path.exists(leftover):
            os.remove(leftover)
        mixed = dict(state) if isinstance(state, dict) else {}
        mixed.update(tensors)
        if mixed:
            import torch
            torch.save(mixed, OUT)
            log("the fp32 checkpoint is in place at %s (%d tensors)"
                % (OUT, len(mixed)))
        elif os.path.exists(bin_path):
            try:
                os.replace(bin_path, OUT)
            except OSError:  # staging sits on another filesystem (a tmpfs)
                shutil.copyfile(bin_path, OUT)
                os.remove(bin_path)
            log("the fp32 checkpoint is in place at %s" % OUT)
        else:
            log("FATAL: neither the safetensors nor the checkpoint could be written")
            return 1
        shutil.rmtree(stage, ignore_errors=True)
        with open(os.path.join(MODEL_DIR, ".weights_source"), "w") as f:
            f.write("Rostlab/prot_bert @ 7a894481acdc12202f0a415dd567f6cfdb698908\n"
                    "fp32 pytorch_model.bin sha256 %s\n"
                    "shipped as fp32: the build-time bf16 conversion failed\n" % PINNED_SHA256)
        log("weights in place at %s" % OUT)
        return 0

    shutil.rmtree(stage, ignore_errors=True)
    digest = file_digest(SAFETENSORS, "sha256")
    with open(os.path.join(MODEL_DIR, ".weights_source"), "w") as f:
        f.write("Rostlab/prot_bert @ 7a894481acdc12202f0a415dd567f6cfdb698908\n"
                "fp32 pytorch_model.bin sha256 %s\n"
                "bf16 model.safetensors sha256 %s\n" % (PINNED_SHA256, digest))
    log("model.safetensors: %d tensors (%d float), %d bytes, sha256 %s"
        % (n_tensors, n_float, os.path.getsize(SAFETENSORS), digest))
    log("weights in place at %s" % SAFETENSORS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
