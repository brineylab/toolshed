"""fetch_files.py: build-time downloads that survive a throttled, dying, or rate-limited network.

Downloads go over plain HTTPS through the Hugging Face resolve URLs' redirect to the CDN
bridge (us.aws.cdn.hf.co), which serves anonymous readers, where hub 2.3.0's
snapshot_download pulls Xet-backed files through the hf-xet client whose anonymous path
this lab's builder is rate-limited on (two builds of this tool died in that step, no file
of the five arriving). The bridge honours bounded Range requests, verified against
biohub/ESMC-600M's model.safetensors.

Modes:

  fetch_files.py BASE_URL OUT_DIR NAME...
      small files in full, each retried with growing backoff and resumed from its partial
      .part file, renamed into place only once its byte count matches the server's
  fetch_files.py --slice K N --size TOTAL URL OUT
      byte range K of N near-equal ranges of the TOTAL-byte file at URL: exactly
      stop - start bytes land at OUT, retried and resumed until they do
  fetch_files.py --assemble N --size TOTAL [--sha256 HEX] PREFIX OUT
      concatenate PREFIX.0 .. PREFIX.N-1 into OUT, check the byte count and (when given)
      the sha256, then delete the slices

The slice mode exists because the builder's builds die mid-download: the fifth build of
this tool's 1.1.0 candidate died 1.25 GiB into the 2.14 GiB model.safetensors, well past
the small files, where one bounded range finishes or stalls inside the read timeout
instead of hanging the build. The Containerfile fetches all the slices and assembles
them inside one RUN, so the slices exist only inside that step and never reach the
image, and the assembled file's sha256 is checked against the repository's LFS oid
before it becomes the image's weights.
"""
import hashlib
import os
import sys
import time
import urllib.request

ATTEMPTS = 4               # per file or slice before the step gives up; a later build resumes
BACKOFF_CAP = 15           # seconds; the sleep before retry n is min(4 * n, this)
CHUNK = 1 << 20            # bytes per read
TIMEOUT = 45               # seconds without a byte from the server before an attempt is dead
PROGRESS_EVERY = 64 << 20  # bytes between progress lines
READ_ASSEMBLE = 4 << 20    # bytes per read while concatenating slices


def note(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def expected_total(response, resumed_at):
    """The file's full size: Content-Range's total when resuming, else Content-Length."""
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    length = response.headers.get("Content-Length")
    if length is None:
        return None
    return resumed_at + int(length) if resumed_at else int(length)


def fetch(base, out_dir, name):
    """One small file, whole: retried, resumed, and renamed in only at its full size."""
    dest = os.path.join(out_dir, name)
    if os.path.exists(dest):
        print(f"{name}: already in place", flush=True)
        return
    part = dest + ".part"
    for attempt in range(1, ATTEMPTS + 1):
        start = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            headers = {"Range": f"bytes={start}-"} if start else {}
            request = urllib.request.Request(base + name, headers=headers)
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                if start and response.getcode() == 200:  # the range was ignored: start over
                    start = 0
                total = expected_total(response, start)
                mode = "ab" if start else "wb"
                got, last_note = start, 0
                with open(part, mode) as f:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        if got - last_note >= PROGRESS_EVERY:
                            note(f"  {name}: {got / (1 << 20):.0f} MiB")
                            last_note = got
            if total is not None and got != total:
                raise IOError(f"short read: {got} of {total} bytes")
            os.replace(part, dest)
            print(f"{name}: {got} bytes", flush=True)
            return
        except Exception as e:
            print(f"{name}: attempt {attempt} failed: {type(e).__name__}: {e}"[:220], flush=True)
            if attempt < ATTEMPTS:
                time.sleep(min(4 * attempt, BACKOFF_CAP))
    sys.exit(f"fetch_files: {name} failed after {ATTEMPTS} attempts")


def slice_bounds(total, n, k):
    """The byte range [start, stop) of slice k when total bytes split into n near-equal parts."""
    if n < 1 or total < 1 or not 0 <= k < n:
        sys.exit(f"fetch_files: no slice {k} of {n} in {total} bytes")
    size = -(-total // n)
    start = k * size
    return start, min(total, start + size)


def fetch_slice(url, out, start, stop):
    """Bytes [start, stop) of url into out, resuming the partial .part file after a dropped
    connection, until exactly stop - start bytes are in place."""
    expected = stop - start
    label = os.path.basename(out)
    if os.path.exists(out):
        if os.path.getsize(out) == expected:
            print(f"{label}: already in place ({expected} bytes)", flush=True)
            return
        sys.exit(f"fetch_files: {out} exists with {os.path.getsize(out)} bytes, expected {expected}")
    part = out + ".part"
    for attempt in range(1, ATTEMPTS + 1):
        got = os.path.getsize(part) if os.path.exists(part) else 0
        if got > expected:
            os.remove(part)
            got = 0
        if got == expected:
            os.replace(part, out)
            print(f"{label}: {expected} bytes in place", flush=True)
            return
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start + got}-{stop - 1}"})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                code = response.getcode()
                if code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    served = content_range.split(" ", 1)[-1].split("/", 1)[0] if content_range else ""
                    if not served.startswith(f"{start + got}-"):
                        raise IOError(f"server answered a different range: {content_range}")
                elif start + got:  # the range was ignored and this slice is not the file's start
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
                        if got - last_note >= PROGRESS_EVERY:
                            note(f"  {label}: {got / (1 << 20):.0f} of {expected / (1 << 20):.0f} MiB")
                            last_note = got
                if got != expected:
                    raise IOError(f"short read: {got} of {expected} bytes")
            os.replace(part, out)
            print(f"{label}: bytes {start}-{stop - 1}, {expected} bytes in place", flush=True)
            return
        except Exception as e:
            print(f"{label}: attempt {attempt} failed: {type(e).__name__}: {e}"[:220], flush=True)
            if attempt < ATTEMPTS:
                time.sleep(min(4 * attempt, BACKOFF_CAP))
    sys.exit(f"fetch_files: {label} failed after {ATTEMPTS} attempts")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(READ_ASSEMBLE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def assemble(prefix, n, out, total, sha256):
    """Concatenate prefix.0 .. prefix.n-1 into out, check size and hash, delete the slices."""
    if os.path.exists(out) and os.path.getsize(out) == total:
        got = sha256_of(out) if sha256 else None
        if sha256 and got != sha256:
            sys.exit(f"fetch_files: {out} hashes to {got}, expected {sha256}")
        print(f"{out}: already assembled ({total} bytes)", flush=True)
    else:
        written, digest = 0, hashlib.sha256()
        with open(out, "wb") as w:
            for k in range(n):
                path = f"{prefix}.{k}"
                start, stop = slice_bounds(total, n, k)
                if not os.path.exists(path):
                    sys.exit(f"fetch_files: {path} is missing; fetch slice {k} first")
                if os.path.getsize(path) != stop - start:
                    sys.exit(f"fetch_files: {path} has {os.path.getsize(path)} bytes, expected {stop - start}")
                with open(path, "rb") as r:
                    while True:
                        chunk = r.read(READ_ASSEMBLE)
                        if not chunk:
                            break
                        digest.update(chunk)
                        w.write(chunk)
                        written += len(chunk)
        if written != total:
            sys.exit(f"fetch_files: assembled {written} of {total} bytes from {n} slices")
        if sha256:
            got = digest.hexdigest()
            if got != sha256:
                sys.exit(f"fetch_files: assembled file hashes to {got}, expected {sha256}")
            print(f"assembled {out}: {total} bytes, sha256 verified", flush=True)
        else:
            print(f"assembled {out}: {total} bytes", flush=True)
    for k in range(n):
        for path in (f"{prefix}.{k}", f"{prefix}.{k}.part"):
            if os.path.exists(path):
                os.remove(path)
    try:
        os.rmdir(os.path.dirname(prefix) or ".")
    except OSError:
        pass


USAGE = ("usage: fetch_files.py BASE_URL OUT_DIR NAME...\n"
         "       fetch_files.py --slice K N --size TOTAL URL OUT\n"
         "       fetch_files.py --assemble N --size TOTAL [--sha256 HEX] PREFIX OUT")


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(USAGE)
    if a[0] == "--slice":
        if len(a) != 7 or a[3] != "--size" or not a[1].isdigit() or not a[2].isdigit() or not a[4].isdigit():
            sys.exit(USAGE)
        k, n, total, url, out = int(a[1]), int(a[2]), int(a[4]), a[5], a[6]
        start, stop = slice_bounds(total, n, k)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        note(f"slice {k + 1} of {n}: bytes {start}-{stop - 1} of {total}")
        fetch_slice(url, out, start, stop)
    elif a[0] == "--assemble":
        if len(a) < 6 or not a[1].isdigit() or a[2] != "--size" or not a[3].isdigit():
            sys.exit(USAGE)
        n, total, rest = int(a[1]), int(a[3]), a[4:]
        sha256 = None
        if rest and rest[0] == "--sha256":
            if len(rest) < 4:
                sys.exit(USAGE)
            sha256, rest = rest[1], rest[2:]
        if len(rest) != 2:
            sys.exit(USAGE)
        assemble(rest[0], n, rest[1], total, sha256)
    elif a[0].startswith("--"):
        sys.exit(USAGE)
    else:
        if len(a) < 3:
            sys.exit(USAGE)
        base, out_dir, names = a[0], a[1], a[2:]
        os.makedirs(out_dir, exist_ok=True)
        print(f"fetching {len(names)} files from {base} into {out_dir}", flush=True)
        for name in names:
            fetch(base, out_dir, name)
        print(f"{len(names)} files in place in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
