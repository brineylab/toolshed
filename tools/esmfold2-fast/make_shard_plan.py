#!/usr/bin/env python3
"""make_shard_plan.py: compute build_shards.py's plan from a revision's safetensors files, downloaded once.

  make_shard_plan.py --tool T --repo R --revision V --dir D --convert-prefix esmc. --out shard_plan.json

D holds the revision's model-*.safetensors, already downloaded. Each is checked against the Hub's LFS
sha256 and size for that revision, then cut at tensor boundaries into spans of at most --budget bytes (a
tensor larger than the budget is a span of its own), and each span's sha256 is taken over those verified
bytes. The plan records the spans, every tensor's place in its file, and the weight map from tensor to the
shard build_shards.py writes it into. Run once, on the build machine; commit only the plan.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

READ = 1 << 22


def sha256_of(path, start=None, end=None):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start or 0)
        left = None if end is None else end - (start or 0)
        while left is None or left > 0:
            chunk = f.read(READ if left is None else min(READ, left))
            if not chunk:
                break
            digest.update(chunk)
            if left is not None:
                left -= len(chunk)
    return digest.hexdigest()


def read_header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return n, json.loads(f.read(n))


def cut(path, budget):
    """The file's tensors in data order, grouped into contiguous spans of at most budget bytes."""
    n, header = read_header(path)
    base = 8 + n
    tensors = sorted(
        ({"name": k, "dtype": v["dtype"], "shape": v["shape"], "offset": base + v["data_offsets"][0],
          "length": v["data_offsets"][1] - v["data_offsets"][0]} for k, v in header.items() if k != "__metadata__"),
        key=lambda t: t["offset"])
    spans, current = [], []
    for t in tensors:
        if current and t["offset"] + t["length"] - current[0]["offset"] > budget:
            spans.append(current)
            current = []
        if current and t["offset"] != current[-1]["offset"] + current[-1]["length"]:
            sys.exit(f"make_shard_plan: {path}: {t['name']} is not contiguous with the tensor before it")
        current.append(t)
    if current:
        spans.append(current)
    return n, header.get("__metadata__"), tensors, spans


def make_plan(tool, repo, revision, directory, prefix, budget, lfs):
    files, raw = [], []
    for name in sorted(lfs):
        path = os.path.join(directory, name)
        size, sha = lfs[name]
        if os.path.getsize(path) != size or sha256_of(path) != sha:
            sys.exit(f"make_shard_plan: {path} does not match the Hub's LFS size and sha256 for {revision}")
        header_len, metadata, tensors, spans = cut(path, budget)
        files.append({"name": name, "size": size, "sha256": sha, "sha256_source": "hub-lfs-oid",
                      "header_len": header_len, "metadata": metadata, "tensors": len(tensors)})
        raw += [(name, path, span) for span in spans]
    total = len(raw)
    shards, weight_map = [], {}
    for i, (name, path, span) in enumerate(raw):
        start, end = span[0]["offset"], span[-1]["offset"] + span[-1]["length"]
        out = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
        shards.append({"index": i, "file": name, "start": start, "end": end,
                       "sha256": sha256_of(path, start, end), "out": out, "tensors": span})
        weight_map.update({t["name"]: out for t in span})
    return {"tool": tool, "repo": repo, "revision": revision,
            "base_url": f"https://huggingface.co/{repo}/resolve/{revision}",
            "dtype_out": "bfloat16", "convert_prefix": prefix, "span_budget_bytes": budget,
            "files": files, "shards": shards, "weight_map": weight_map,
            "total_size": sum(t["length"] for s in shards for t in s["tensors"])}


def hub_lfs(repo, revision):
    """{file name: (size, sha256)} for the revision's safetensors shards, from the Hub's own LFS records."""
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true") as r:
        info = json.load(r)
    return {s["rfilename"]: (s["lfs"]["size"], s["lfs"]["sha256"]) for s in info["siblings"]
            if s["rfilename"].endswith(".safetensors")}


def selftest():
    """Three fp32 tensors of 400, 400, and 2000 bytes under a 1000-byte budget cut into two spans that
    cover every tensor once, contiguous, each hashed over its own bytes."""
    import tempfile

    import torch
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model-00001-of-00001.safetensors")
        save_file({"esmc.a": torch.zeros(100), "esmc.b": torch.ones(100), "trunk.c": torch.ones(500)}, path)
        plan = make_plan("t", "r", "v", d, "esmc.", 1000, {os.path.basename(path): (os.path.getsize(path), sha256_of(path))})
        spans = [[t["name"] for t in s["tensors"]] for s in plan["shards"]]
        assert spans == [["esmc.a", "esmc.b"], ["trunk.c"]], spans
        for s in plan["shards"]:
            assert s["end"] - s["start"] == sum(t["length"] for t in s["tensors"]), s
            assert s["sha256"] == sha256_of(path, s["start"], s["end"]), s
        assert plan["weight_map"] == {"esmc.a": "model-00001-of-00002.safetensors",
                                      "esmc.b": "model-00001-of-00002.safetensors",
                                      "trunk.c": "model-00002-of-00002.safetensors"}, plan["weight_map"]
        assert plan["total_size"] == 2800, plan["total_size"]
    print("selftest: ok")


def main():
    if sys.argv[1:] == ["--selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(prog="make_shard_plan", description=__doc__.splitlines()[0])
    ap.add_argument("--tool", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--convert-prefix", required=True)
    ap.add_argument("--budget", type=int, default=256 << 20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = make_plan(a.tool, a.repo, a.revision, a.dir, a.convert_prefix, a.budget, hub_lfs(a.repo, a.revision))
    with open(a.out, "w") as f:
        json.dump(plan, f, indent=1)
        f.write("\n")
    print(f"{a.out}: {len(plan['shards'])} shards over {len(plan['files'])} files, {plan['total_size']} bytes of tensors")


if __name__ == "__main__":
    main()
