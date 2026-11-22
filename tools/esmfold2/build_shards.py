#!/usr/bin/env python3
"""build_shards.py: build one shard of a tool's weights from its shard plan, or the index.

esmc-6b's builder, adapted: the plan (shard_plan.json, committed beside this file, made by
make_shard_plan.py from the pinned revision's own safetensors files) cuts them at tensor boundaries into
spans of at most span_budget_bytes, records every span's sha256 over bytes the Hub's LFS oid verified, and
the whole tensor table. This script builds the image's weights from it one bounded step at a time, since
esmc-6b's builds died in steps that moved about a GiB or more while steps of at most 256 MiB survived, and
completed RUNs replay from the layer cache, so a build that dies resumes at its first missing shard:

  build_shards.py --plan P --shard K
      fetch shard K's span over the repository's resolve URL, retried with backoff and resumed from its
      .part file until exactly the span's bytes arrive and hash to the plan's sha256; write them, header
      first, as model-XXXXX-of-XXXXX.safetensors; re-read it and verify its tensors' sizes and its data
      section's sha256. Then convert it in place: tensors named under the plan's convert_prefix (the ESM C
      backbone, "esmc.") are cast to bf16 (round-to-nearest-even), the rest written unchanged, and the
      converted shard's structure verified; the fp32 bytes never leave the step, so they never reach a layer.
  build_shards.py --plan P --index
      write model.safetensors.index.json from the plan's weight map.
  build_shards.py --selftest
      convert a two-tensor shard under the prefix rule and check the dtypes it leaves.

config.json is left as published. Every step prints what it did, with timestamps. Nothing here reaches the
network at run time; the image sets HF_HUB_OFFLINE.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

ATTEMPTS = 4               # per shard before the step gives up; a later build resumes here
BACKOFF_CAP = 15           # seconds; the sleep before retry n is min(4 * n, this)
CHUNK = 1 << 20            # bytes per read
CONNECTIONS = 8           # concurrent ranges per span
TIMEOUT = 45               # seconds without a byte from the server before an attempt is dead
READ = 1 << 22             # bytes per read while hashing or copying
ELEM = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def note(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def sha256_of(path, start=None, end=None):
    """The sha256 of the whole file, or of its bytes [start, end)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        if start is not None:
            f.seek(start)
        while True:
            chunk = f.read(READ if end is None else min(READ, end - f.tell()))
            if not chunk:
                break
            digest.update(chunk)
            if end is not None and f.tell() >= end:
                break
    return digest.hexdigest()


def fetch_span(url, part, start, end):
    """Bytes [start, end) of url into part, resuming the partial .part after a dropped
    connection, until exactly end - start bytes are in place. As fetch_files.py's slices,
    which this builder served correctly nine times over."""
    expected = end - start
    label = os.path.basename(part)
    for attempt in range(1, ATTEMPTS + 1):
        got = os.path.getsize(part) if os.path.exists(part) else 0
        if got > expected:
            os.remove(part)
            got = 0
        if got == expected:
            return
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start + got}-{end - 1}"})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                code = response.getcode()
                if code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    served = content_range.split(" ", 1)[-1].split("/", 1)[0] if content_range else ""
                    if not served.startswith(f"{start + got}-"):
                        raise IOError(f"server answered a different range: {content_range}")
                elif start + got:  # the range was ignored and this span is not the file's start
                    raise IOError("server ignored the range request")
                mode = "ab" if got else "wb"
                last_note = 0
                with open(part, mode) as f:
                    while got < expected:
                        chunk = response.read(min(CHUNK, expected - got))
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        if got - last_note >= 64 << 20:
                            note(f"  {label}: {got / (1 << 20):.0f} of {expected / (1 << 20):.0f} MiB")
                            last_note = got
                if got != expected:
                    raise IOError(f"short read: {got} of {expected} bytes")
            return
        except Exception as e:
            print(f"{label}: attempt {attempt} failed: {type(e).__name__}: {e}"[:220], flush=True)
            if attempt < ATTEMPTS:
                time.sleep(min(4 * attempt, BACKOFF_CAP))
    sys.exit(f"build_shards: {label} failed after {ATTEMPTS} attempts")


def subranges(start, end, n):
    """[start, end) cut into n contiguous near-equal ranges."""
    cuts = [start + (end - start) * k // n for k in range(n + 1)]
    return list(zip(cuts, cuts[1:]))


def fetch_span_parallel(url, part, start, end):
    """fetch_span over CONNECTIONS ranges at once, concatenated into part: one connection to the CDN can
    run at under 1 MB/s while many run at tens."""
    from concurrent.futures import ThreadPoolExecutor

    pieces = subranges(start, end, CONNECTIONS)
    with ThreadPoolExecutor(CONNECTIONS) as pool:
        list(pool.map(lambda kp: fetch_span(url, f"{part}.{kp[0]}", *kp[1]), enumerate(pieces)))
    with open(part, "wb") as w:
        for k in range(len(pieces)):
            with open(f"{part}.{k}", "rb") as r:
                while True:
                    chunk = r.read(READ)
                    if not chunk:
                        break
                    w.write(chunk)
            os.remove(f"{part}.{k}")


def shard_header_bytes(file_entry, shard):
    """The shard's safetensors header: __metadata__ carried over from the repository's
    file, each tensor's offsets rebased to the span's start, space-padded to 8 bytes."""
    header = {}
    if file_entry.get("metadata") is not None:
        header["__metadata__"] = file_entry["metadata"]
    start = shard["start"]
    for t in shard["tensors"]:
        header[t["name"]] = {"dtype": t["dtype"], "shape": t["shape"],
                             "data_offsets": [t["offset"] - start, t["offset"] - start + t["length"]]}
    js = json.dumps(header, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 8)
    return len(js).to_bytes(8, "little") + js


def verify_fp32_shard(path, file_entry, shard):
    """Re-read the written shard: its header parses, its tensors are the plan's sizes,
    and its data section hashes to the span's sha256."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    if header.get("__metadata__") != file_entry.get("metadata"):
        sys.exit(f"build_shards: {path}: metadata changed under us")
    reach = 8 + n
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        start, end = spec["data_offsets"]
        if spec["dtype"] not in ELEM or end - start != _prod(spec["shape"]) * ELEM[spec["dtype"]]:
            sys.exit(f"build_shards: {path}: {name} is {end - start} bytes, "
                     f"{spec['shape']} x {spec['dtype']} disagrees")
        if 8 + n + start < reach:
            sys.exit(f"build_shards: {path}: {name} overlaps the tensor before it")
        reach = 8 + n + end
    size = os.path.getsize(path)
    if reach > size:
        sys.exit(f"build_shards: {path}: header reaches {reach} bytes, file is {size}")
    if sha256_of(path, 8 + n, size) != shard["sha256"]:
        sys.exit(f"build_shards: {path}: data section does not hash to the plan's span sha256")


def _prod(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def verify_converted_shard(path, prefix):
    """The converted shard's structure: its header parses, every tensor under the prefix is BF16 of its
    shape, and every other tensor is its own dtype of its shape."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    reach = 8 + n
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        start, end = spec["data_offsets"]
        if name.startswith(prefix) and spec["dtype"] != "BF16":
            sys.exit(f"build_shards: {path}: {name} is {spec['dtype']}, not bf16")
        if spec["dtype"] not in ELEM or end - start != _prod(spec["shape"]) * ELEM[spec["dtype"]]:
            sys.exit(f"build_shards: {path}: {name} is {end - start} bytes, {spec['shape']} x {spec['dtype']} disagrees")
        if 8 + n + start < reach:
            sys.exit(f"build_shards: {path}: {name} overlaps the tensor before it")
        reach = 8 + n + end
    if reach > os.path.getsize(path):
        sys.exit(f"build_shards: {path}: header reaches past the file")


def convert_shard(src, out, prefix):
    """Write src's tensors to out, those named under prefix cast to bf16 (round-to-nearest-even), the rest
    as they are. Returns the bytes of tensor data written."""
    import torch
    from safetensors.torch import load_file, save_file

    tensors = {k: v.to(torch.bfloat16) if k.startswith(prefix) else v for k, v in load_file(src).items()}
    save_file(tensors, out, metadata={"format": "pt"})
    return sum(v.numel() * v.element_size() for v in tensors.values())


def build_shard(plan, index, model_dir):
    shard = plan["shards"][index]
    file_entry = next(f for f in plan["files"] if f["name"] == shard["file"])
    out = os.path.join(model_dir, shard["out"])
    if os.path.exists(out):
        if plan["dtype_out"] == "bfloat16":
            verify_converted_shard(out, plan["convert_prefix"])
        else:
            verify_fp32_shard(out, file_entry, shard)
        note(f"shard {index + 1}/{len(plan['shards'])}: {shard['out']} already in place and verified")
        return
    os.makedirs(os.path.join(model_dir, ".parts"), exist_ok=True)
    part = os.path.join(model_dir, ".parts", shard["out"] + ".part")
    for attempt in range(1, ATTEMPTS + 1):
        fetch_span_parallel(f"{plan['base_url']}/{shard['file']}", part, shard["start"], shard["end"])
        got = sha256_of(part)
        if got == shard["sha256"]:
            break
        print(f"{shard['out']}: attempt {attempt} hashed to {got}, the plan says {shard['sha256']}; "
              f"discarding and refetching", flush=True)
        os.remove(part)
    else:
        sys.exit(f"build_shards: {shard['out']}: the span never hashed to the plan's sha256")
    header = shard_header_bytes(file_entry, shard)
    tmp = out + ".tmp"
    with open(tmp, "wb") as w:
        w.write(header)
        with open(part, "rb") as r:
            while True:
                chunk = r.read(READ)
                if not chunk:
                    break
                w.write(chunk)
    verify_fp32_shard(tmp, file_entry, shard)
    if plan["dtype_out"] == "bfloat16":
        converted = tmp + ".bf16"
        written = convert_shard(tmp, converted, plan["convert_prefix"])
        os.remove(tmp)
        os.replace(converted, out)
        verify_converted_shard(out, plan["convert_prefix"])
        note(f"shard {index + 1}/{len(plan['shards'])}: {shard['out']}: span {shard['end'] - shard['start']} "
             f"bytes fetched, sha256 verified, {plan['convert_prefix']}* converted to bf16 "
             f"({written} bytes) and verified")
    else:
        os.replace(tmp, out)
        note(f"shard {index + 1}/{len(plan['shards'])}: {shard['out']}: span {shard['end'] - shard['start']} "
             f"bytes fetched, sha256 verified, written and re-verified")
    os.remove(part)
    try:
        os.rmdir(os.path.join(model_dir, ".parts"))
    except OSError:
        pass


def write_index(plan, model_dir):
    """model.safetensors.index.json from the plan: the layout from_pretrained loads."""
    total = 0
    for shard in plan["shards"]:
        for t in shard["tensors"]:
            converted = plan["dtype_out"] == "bfloat16" and t["name"].startswith(plan["convert_prefix"])
            stored = t["length"] // ELEM[t["dtype"]] * 2 if converted else t["length"]
            total += stored
    index = {"metadata": {"total_size": total}, "weight_map": plan["weight_map"]}
    path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(path, "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
    note(f"index: {len(plan['weight_map'])} tensors over {len(plan['shards'])} shards, "
         f"{total} bytes of weights, written to {path}")


def main():
    ap = argparse.ArgumentParser(prog="build-shards", description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--shard", type=int)
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--model-dir", default=os.environ.get("ESMFOLD2_MODEL", "/opt/model"))
    args = ap.parse_args()
    plan = json.load(open(args.plan))
    if (args.shard is not None) == args.index:
        sys.exit("build_shards: give exactly one of --shard K, --index")
    if args.shard is not None:
        if not 0 <= args.shard < len(plan["shards"]):
            sys.exit(f"build_shards: no shard {args.shard} of {len(plan['shards'])}")
        build_shard(plan, args.shard, args.model_dir)
    else:
        write_index(plan, args.model_dir)


def selftest():
    """A two-tensor fp32 shard, converted under the prefix rule: esmc.x becomes BF16, trunk.y stays F32."""
    import tempfile

    import torch
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "in.safetensors"), os.path.join(d, "out.safetensors")
        save_file({"esmc.x": torch.ones(3, 4), "trunk.y": torch.full((5,), 1.0 / 3)}, src, metadata={"format": "pt"})
        convert_shard(src, out, "esmc.")
        with open(out, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        assert header["esmc.x"]["dtype"] == "BF16", header
        assert header["trunk.y"]["dtype"] == "F32", header
        verify_converted_shard(out, "esmc.")
    pieces = subranges(100, 1101, 8)
    assert pieces[0][0] == 100 and pieces[-1][1] == 1101, pieces
    assert all(a[1] == b[0] for a, b in zip(pieces, pieces[1:])), pieces
    assert len(pieces) == 8 and max(e - s for s, e in pieces) - min(e - s for s, e in pieces) <= 1, pieces
    print("selftest: ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        main()
