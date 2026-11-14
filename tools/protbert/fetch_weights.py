#!/usr/bin/env python3
"""Put ProtBERT's weights in /opt/model as one bf16 safetensors file, pinned by sha256.

Four builds of this image died in the weights step: on the machine that
builds them, connections to huggingface.co open and then move no bytes --
for the 1.7 GB checkpoint and the 81-byte vocab.txt alike -- while PyPI,
the base image and ghcr.io all pull normally, and the cluster fetches the
same repo in half a minute. So the checkpoint is taken (in the base image's
own python, nothing new installed, no curl, no apt) from the first of two
sources that can deliver it:

  1. the pinned huggingface.co URL, the same bytes every sibling tool uses;
  2. ProtTrans's own Zenodo record 4633691 (the project's prot_bert.zip),
     a different network end to end.

and ships only when its sha256 is the pinned one, whichever source
delivered the bytes. Every event goes to stderr with a timestamp, so a
build log says exactly which source, attempt and byte count failed.

The fp32 checkpoint is then converted, in this same step, to a bf16
safetensors file and deleted, so the layer the image carries holds the
0.84 GB the computation wants rather than the 1.68 GB fp32 original: the
same bf16 the wrapper would otherwise cast to at load time, so the weights
are bit for bit what a bf16 load of the fp32 checkpoint gives, at half the
bytes to build, push and pull.

The tokenizer and config files are not fetched: the four of them, and the
license, are 11 KB in all and are committed in tools/protbert/model/
(their sha256s are checked below), so the network moves one file, the
checkpoint.
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
TMP_DIR = os.path.join(MODEL_DIR, ".downloads")

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
        "deadline_s": 6 * 60,
    },
    {
        "name": "zenodo.org record 4633691, ProtTrans's prot_bert.zip",
        "url": "https://zenodo.org/api/records/4633691/files/prot_bert.zip/content",
        "kind": "zip",
        "size": 1561909078,  # the record's own file listing, with its md5
        "hash": ("md5", "1bc4f9ae39e0d38bb43ecf049337893f"),
        "deadline_s": 30 * 60,
    },
]

READ_TIMEOUT_S = 60  # a read that moves no bytes for a minute is a stalled connection
CHUNK = 1 << 20


def log(message):
    print("[fetch_weights %s] %s" % (time.strftime("%H:%M:%S"), message), file=sys.stderr, flush=True)


def file_digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


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
                log("attempt %d: HTTP %s from %s, %d bytes already present, content-length %s"
                    % (attempt, r.status, host, have, r.headers.get("Content-Length")))
                with open(path, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        now = time.monotonic()
                        if now - noted >= 15:
                            log("attempt %d: %d of %d bytes, %.1f MB/s"
                                % (attempt, have + got, expected_size, got / (now - started) / 1e6))
                            noted = now
            now = os.path.getsize(path)
            log("attempt %d: read %d bytes; the file now holds %d of %d" % (attempt, got, now, expected_size))
            if now == expected_size:
                return True
            log("short read; retrying with resume")
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout,
                TimeoutError, ConnectionError, OSError) as e:
            log("attempt %d failed: %s: %s" % (attempt, type(e).__name__, e))
            time.sleep(min(10, max(0.0, deadline - time.monotonic())))
    return False


def extract_bin(zip_path):
    """Pull pytorch_model.bin out of ProtTrans's prot_bert.zip into OUT."""
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
        with zf.open(member) as src, open(OUT, "wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)
    return True


def to_bf16_safetensors(bin_path, out_path):
    """The checkpoint as one bf16 safetensors file: every floating tensor
    cast to bf16 (the cast a `dtype=torch.bfloat16` load would do), every
    other tensor as it is. Returns (tensors, float tensors)."""
    import torch
    from safetensors.torch import save_file

    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("the checkpoint is not a state dict: %s" % type(state))
    tensors, n_float = {}, 0
    for key in sorted(state):
        tensor = state.pop(key)  # pop, so the fp32 dict shrinks as the bf16 one grows
        if not torch.is_tensor(tensor):
            raise ValueError("%s is a %s, not a tensor" % (key, type(tensor)))
        if torch.is_floating_point(tensor):
            tensor = tensor.to(torch.bfloat16)
            n_float += 1
        # clone, so no two keys share storage (safetensors refuses that) and
        # the saved tensors hold no reference into the fp32 checkpoint
        tensors[key] = tensor.clone().contiguous()
    save_file(tensors, out_path, metadata={"format": "pt"})
    return len(tensors), n_float


def main():
    log("python %s; model dir %s" % (sys.version.split()[0], MODEL_DIR))
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

    if not (os.path.exists(OUT) and file_digest(OUT, "sha256") == PINNED_SHA256):
        if os.path.exists(OUT):
            log("%s exists but is not the pinned checkpoint; refetching" % OUT)
            os.remove(OUT)
        delivered = False
        os.makedirs(TMP_DIR, exist_ok=True)
        for src in SOURCES:
            log("source: %s" % src["name"])
            tmp = os.path.join(TMP_DIR, "weights.bin" if src["kind"] == "bin" else "prot_bert.zip")
            if not download(src["url"], tmp, src["size"], src["deadline_s"]):
                log("the source could not deliver its file in %d s: %s" % (src["deadline_s"], src["name"]))
                continue
            algo, want = src["hash"]
            got = file_digest(tmp, algo)
            log("%s of the downloaded file: %s (expected %s)" % (algo, got, want))
            if got != want:
                log("hash mismatch; these bytes are not shipped")
                os.remove(tmp)
                continue
            if src["kind"] == "bin":
                os.replace(tmp, OUT)
            else:
                if not extract_bin(tmp):
                    os.remove(tmp)
                    continue
                os.remove(tmp)
            if file_digest(OUT, "sha256") != PINNED_SHA256:
                os.remove(OUT)  # wrong bytes never stay in the model dir
                log("FATAL: the delivered checkpoint does not match the pinned sha256")
                return 1
            delivered = True
            break
        if not delivered:
            log("FATAL: no source delivered the pinned checkpoint")
            return 1
        log("the pinned fp32 checkpoint is in place; converting to bf16")

    n_tensors, n_float = to_bf16_safetensors(OUT, SAFETENSORS)
    os.remove(OUT)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
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
