#!/usr/bin/env python3
"""Convert one group of ESM-2 15B's tensors to bf16, as one short build step.

Builds of these images die on the machine that builds them in or after any step
that moves about a GiB or more, and facebook/esm2_t48_15B_UR50D releases its
weights as seven fp32 shards totalling 60.5 GB -- neither a fetch nor a
conversion of that size can be one step. So per the brief the image ships the
weights converted to bf16 (~30 GB), in ~150 groups of state-dict tensors, and
this step converts group k of manifest.json into one ~192 MiB .bin shard:

- it re-fetches only that group's fp32 byte spans -- one resumable, verified,
  mirror-falling-back ranged GET per tensor -- so the fp32 bytes never persist
  in the image and each step moves well under a GiB;
- each span is checked against the sha256 the assembly validation run pinned
  after bit-verifying it against torch.load(mmap=True) of the assembled,
  whole-sha256-checked shards, and the bf16 shard this step writes is checked
  against the value pin that same run recorded, so the weights are pinned
  twice;
- the cast is the rounding oplm's computation runs on (fp32 to bf16), and the
  shard is torch.save's dict of those tensors, proven by loading it back: it
  must hold exactly those tensors and hash to the group's canonical value
  sha256. torch.save's container bytes are not deterministic -- run 426380dd
  saved one tensors dict four times and got four distinct sha256s, and two
  saves of ESM-2 15B's group 0 even differ in length -- but the tensors every
  save loads are bit-identical, in one process and across processes, so the
  pins pin the values. The one int64 tensor, esm.embeddings.position_ids,
  rides its group unchanged (bf16 cannot hold 1021 exactly, and the released
  checkpoint carries it too);
- the spans are fetched to a temporary file that never outlives the step, and
  the only layer the step commits is its one shard.

The group index k is this step's place in the Containerfile's RUN instructions,
which never change between builds, so whatever a failed build leaves in the
builder's layer cache the next build reuses.

    python3 -I convert_step.py <index>
"""
import hashlib
import json
import os
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
import torch

REVISION = "5fbca39631164edc1d402a5aa369f982f72ee282"
REPO = "facebook/esm2_t48_15B_UR50D"
MODEL_DIR = os.environ.get("ESM2_15B_OUT", "/opt/model")
MANIFEST_PATH = os.environ.get(
    "ESM2_15B_MANIFEST",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json"))

# The manifest is read on demand (load_manifest below), not at import: the
# assembly validation run imports this file for group_value_sha256 before the
# manifest exists -- it is the run whose output becomes the manifest.

SOURCES = [
    ("huggingface.co", "https://huggingface.co/%s/resolve/%s/%%s" % (REPO, REVISION)),
    ("hf-mirror.com", "https://hf-mirror.com/%s/resolve/%s/%%s" % (REPO, REVISION)),
]

# Every dtype the checkpoint holds: the weights are fp32 and cast to bf16 here, and
# esm.embeddings.position_ids is an int64 buffer that rides its group unchanged --
# bf16 cannot hold 1021 exactly, and the released checkpoint carries it too, so the
# image's model dir loads exactly as the fp32 original does.
ROW_NP_DTYPE = {"float32": "<f4", "int64": "<i8"}

READ_TIMEOUT_S = 20    # a read that moves no bytes for this long is a stalled connection
STEP_DEADLINE_S = 240  # this step fails, having owed the builder only this much
BLOCK = 1 << 20


def log(message):
    print("[convert_step %s] %s" % (time.strftime("%H:%M:%S"), message), file=sys.stderr, flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def group_value_sha256(tensors, keys):
    """The group's canonical value pin: sha256 over key, dtype, shape, raw bits.

    torch.save's zip container is not byte-deterministic -- run 426380dd saved
    one tensors dict four times and got four distinct sha256s (two saves of
    ESM-2 15B's group 0 even differ in length), while every save loads to
    bit-identical tensors, in one process and across processes -- so the pins
    are pins on the values. bf16 bits hash as uint16; the group's rows fix the
    key order.
    """
    h = hashlib.sha256()
    for key in keys:
        t = tensors[key]
        if not isinstance(t, torch.Tensor) or not t.is_contiguous():
            raise ValueError("%s: not a contiguous tensor" % key)
        h.update(key.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(":".join(str(d) for d in t.shape).encode("utf-8"))
        h.update(t.view(torch.uint16).numpy().tobytes() if t.dtype == torch.bfloat16
                 else t.numpy().tobytes())
    return h.hexdigest()


def shard_value_sha256(path, keys):
    """The value pin of a shard already on disk, or None when it will not load."""
    try:
        loaded = torch.load(path, mmap=True, weights_only=True)
        if list(loaded.keys()) != keys:
            return None
        return group_value_sha256(loaded, keys)
    except Exception:
        return None


def load_manifest():
    """The tree's manifest: the seven released fp32 shards and the groups to convert."""
    manifest = json.load(open(MANIFEST_PATH))
    return manifest["fp32_shards"], manifest["groups"]


def fetch_once(name, url, start, length, want, target):
    """One ranged attempt from one source. True when target holds the pinned bytes."""
    have = os.path.getsize(target) if os.path.exists(target) else 0
    if have > length:
        log("%s: %d bytes held is past the span's %d; starting over" % (name, have, length))
        os.remove(target)
        have = 0
    end = start + length - 1
    headers = {"User-Agent": "brineylab-toolshed-esm2-15b-build",
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
        log("%s: span bytes %d-%d, %d already held, content-length %s"
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
        log("%s: the span hashes to %s, not the pinned %s; refetching it from byte 0"
            % (name, digest, want))
        os.remove(target)
        return False
    log("%s: span delivered and verified in %.1f s" % (name, time.monotonic() - began))
    return True


def fetch_span_bytes(row, fp32_shards):
    """One tensor's fp32 span: resumable, sha256-verified, falling back to the mirror."""
    shard = fp32_shards[row["shard"]]
    assert 0 <= row["offset"] and row["offset"] + row["length"] <= shard["size"], row["key"]
    fd, target = tempfile.mkstemp(prefix=".span-")
    os.close(fd)
    try:
        deadline = time.monotonic() + STEP_DEADLINE_S
        attempt = 0
        while time.monotonic() < deadline:
            name, template = SOURCES[attempt % len(SOURCES)]
            attempt += 1
            try:
                if fetch_once("%s %s" % (shard["name"], row["key"]), template % shard["name"],
                              row["offset"], row["length"], row["sha256"], target):
                    with open(target, "rb") as f:
                        return f.read()
            except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout,
                    TimeoutError, ConnectionError, OSError) as e:
                log("%s: attempt %d (%s) failed: %s: %s"
                    % (row["key"], attempt, name, type(e).__name__, e))
        log("FATAL: the span of %s was not delivered in %d s" % (row["key"], STEP_DEADLINE_S))
        return None
    finally:
        if os.path.exists(target):
            os.remove(target)


def convert(index, groups, fp32_shards):
    group = groups[index]
    keys = [row["key"] for row in group["rows"]]
    out_path = os.path.join(MODEL_DIR, group["name"])
    if os.path.exists(out_path) and shard_value_sha256(out_path, keys) == group["value_sha256"]:
        log("group %d of %d already holds its pinned values; nothing to do"
            % (index, len(groups) - 1))
        return True

    tensors = {}
    for row in group["rows"]:
        raw = fetch_span_bytes(row, fp32_shards)
        if raw is None:
            return False
        dtype = row.get("dtype", "float32")
        t = torch.from_numpy(np.frombuffer(raw, dtype=ROW_NP_DTYPE[dtype]).copy()).reshape(row["shape"])
        assert t.numel() == row["numel"], row["key"]
        tensors[row["key"]] = t.to(torch.bfloat16) if dtype == "float32" else t

    os.makedirs(MODEL_DIR, exist_ok=True)
    # the suffix is load-bearing: torch.save names the zip's inner archive after
    # the file's basename with everything from the last '.' stripped, so a
    # dotfile with no later dot (".group-f8oesc62") gives an empty archive name
    # and PyTorchFileWriter dies with "invalid file name" (run 369e7788);
    # ".group-f8oesc62.bin" leaves ".group-f8oesc62"
    fd, tmp = tempfile.mkstemp(prefix=".group-", suffix=".bin", dir=MODEL_DIR)
    os.close(fd)
    torch.save(tensors, tmp)
    # the proof is a load-back: the file just written must hold exactly the
    # tensors this step converted, hashing to the group's pinned value
    try:
        loaded = torch.load(tmp, mmap=True, weights_only=True)
        assert list(loaded.keys()) == keys, "the saved shard holds other keys"
        for key in keys:
            assert torch.equal(loaded[key], tensors[key]), "the saved %s differs" % key
        pin = group_value_sha256(loaded, keys)
    except Exception as e:
        os.remove(tmp)
        log("FATAL: group %d: the saved shard does not load back: %s: %s"
            % (index, type(e).__name__, e))
        return False
    if pin != group["value_sha256"]:
        os.remove(tmp)
        log("FATAL: group %d pinned to %s, not the manifest's %s; the conversion "
            "changed the values" % (index, pin, group["value_sha256"]))
        return False
    held = os.path.getsize(tmp)
    os.chmod(tmp, 0o644)
    os.replace(tmp, out_path)
    log("group %d of %d: %d keys, %d bytes of bf16, value pin verified"
        % (index, len(groups) - 1, len(group["rows"]), held))
    if index == len(groups) - 1:
        log("the last group: the image now holds all %d shards" % len(groups))
    return True


def main():
    fp32_shards, groups = load_manifest()
    if len(sys.argv) != 2:
        log("usage: convert_step.py <index 0-%d>" % (len(groups) - 1))
        return 2
    try:
        index = int(sys.argv[1])
    except ValueError:
        log("the group index is not a number: %r" % sys.argv[1])
        return 2
    if not 0 <= index < len(groups):
        log("group index %s is outside 0-%d" % (sys.argv[1], len(groups) - 1))
        return 2
    total = sum(len(g["rows"]) for g in groups)
    log("python %s; group %d of %d: %d keys, %d bytes of bf16, %d tensors across all groups"
        % (sys.version.split()[0], index, len(groups) - 1, len(groups[index]["rows"]),
           groups[index]["bytes"], total))
    return 0 if convert(index, groups, fp32_shards) else 1


if __name__ == "__main__":
    sys.exit(main())
