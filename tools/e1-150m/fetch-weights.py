#!/usr/bin/env python3
"""Pull E1's pinned weights into /opt/model, over whichever route the build host can reach.

This image's first two builds died in this step: huggingface.co itself answered
(config.json arrived, with the revision's metadata), but model.safetensors never
arrived, and every retry failed at once.  huggingface_hub 0.36 carries hf-xet as
a hard dependency and sends Xet-backed files through hf-xet's own endpoints, so
the pull now tries three routes, each from a clean state, resuming partial
downloads within a route:

  cdn    plain HTTPS, hf-xet disabled (HF_HUB_DISABLE_XET=1): huggingface.co
         redirects to its file CDN (us.aws.cdn.hf.co for this repository)
  xet    the hf-xet protocol that huggingface_hub uses by default
  curl   plain HTTPS again, through curl rather than requests

Whatever lands is checked against the pinned sha256 before the build continues:
a route that served different bytes fails the build instead of baking them in.
Every failed attempt prints the exception it got, so a build that dies here says
why.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import time

REPO = "Profluent-Bio/E1-150m"
REVISION = "c4dbfe827e4aa6ed7f95eaef50dc1e084f4d77dc"
PATTERNS = ["*.json", "*.safetensors"]
FILES = ("config.json", "model.safetensors")
SHA256 = "ba2656339005e6598642836acfdafde480fecc7e145ce0058eb54adf572c3484"
DEST = "/opt/model"
XET_CACHE = "/tmp/hf-xet-cache"  # hf-xet's chunk cache, kept out of the image

ATTEMPTS = 4           # per route
SLEEP_S = (5, 10, 20)  # between attempts of one route

# The routes, in the order they are tried: (label, the env a fresh
# huggingface_hub import sees; its constants are read at import time, so each
# attempt runs as its own process).
ROUTES = (
    ("cdn", {"HF_HUB_DISABLE_XET": "1"}),
    ("xet", {}),
)


def log(*parts):
    print(*parts, flush=True)


def partial_mib():
    """How far a route got: the MiB held in .incomplete files so far."""
    total = 0
    for root, _dirs, names in os.walk(DEST):
        for name in names:
            if name.endswith(".incomplete"):
                total += os.path.getsize(os.path.join(root, name))
    return total // (1 << 20)


def child_pull(label):
    """The download itself, in a child process; its env lands before the import."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_XET_CACHE"] = XET_CACHE
    for key, value in dict(ROUTES)[label].items():
        os.environ[key] = value

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=REVISION, local_dir=DEST, allow_patterns=PATTERNS)
    except BaseException as exc:  # the parent relays this into the build log
        log(f"PULL-FAILED {type(exc).__name__}: {' '.join(str(exc).split())[:220]}")
        raise SystemExit(1) from None
    log(f"{label}: snapshot_download ok")


def hub_route(label):
    """One huggingface_hub route: ATTEMPTS tries, resuming what the last left."""
    for attempt in range(1, ATTEMPTS + 1):
        log(f"route {label} attempt {attempt}/{ATTEMPTS}")
        done = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--pull", label])
        if done.returncode == 0:
            return True
        log(f"route {label} attempt {attempt} exited {done.returncode} at {partial_mib()} MiB")
        if attempt < ATTEMPTS:
            time.sleep(SLEEP_S[min(attempt, len(SLEEP_S)) - 1])
    return False


def curl_route():
    """The last route: curl instead of requests, straight at the resolve URLs."""
    for name in FILES:
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        target = os.path.join(DEST, name)
        for attempt in range(1, ATTEMPTS + 2):
            log(f"route curl {name} attempt {attempt}")
            done = subprocess.run(
                ["curl", "-fL", "--connect-timeout", "15", "--retry", "3",
                 "--retry-all-errors", "-C", "-", "-o", target, url],
            )
            if done.returncode == 0:
                break
            if attempt > ATTEMPTS:
                return False
            time.sleep(SLEEP_S[-1])
    return True


def verify():
    """The pinned revision's bytes, or no build."""
    for name in FILES:
        if not os.path.isfile(os.path.join(DEST, name)):
            log(f"verify: {name} did not arrive")
            return False
    digest = hashlib.sha256()
    with open(os.path.join(DEST, "model.safetensors"), "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    got = digest.hexdigest()
    if got != SHA256:
        log(f"verify: sha256 {got} is not the pinned {SHA256}")
        return False
    log(f"verify: model.safetensors sha256 {got} == the pinned value")
    return True


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--pull":
        child_pull(sys.argv[2])
        return 0
    log(f"pull {REPO} at {REVISION} into {DEST}: routes cdn, xet, curl")
    tried = []
    try:
        ok = hub_route("cdn")
        if not ok:
            tried.append("cdn")
            shutil.rmtree(DEST, ignore_errors=True)  # no resume state across routes
            ok = hub_route("xet")
        if not ok:
            tried.append("xet")
            shutil.rmtree(DEST, ignore_errors=True)
            ok = curl_route()
        if not ok:
            tried.append("curl")
        if ok:
            ok = verify()
        if not ok:
            log(f"weights pull failed; routes tried: {', '.join(tried) or 'verify'}")
            return 1
        shutil.rmtree(os.path.join(DEST, ".cache"), ignore_errors=True)  # hub bookkeeping
    finally:
        shutil.rmtree(XET_CACHE, ignore_errors=True)
    log("weights in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
