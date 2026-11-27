"""Verify the named checkpoint and cached alignments installed in the image."""
import hashlib
import json
import pathlib

from bioemu2.checkpoints import get_checkpoint_files

f = get_checkpoint_files("bioemu2-preview")
print("checkpoint:", f)
assert str(f.checkpoint_path).startswith("/toolshed/weights/")
assert f.checkpoint_path.stat().st_size > 1000000000
n = 0
for p in pathlib.Path("/opt/bioemu2/msa-cache").glob("*.json"):
    d = json.loads(p.read_text())
    assert d["sha256"] == hashlib.sha256(json.dumps(d["a3ms"]).encode("utf-8")).hexdigest(), p
    n += 1
assert n, "no cached alignments"
print(n, "cached ColabFold alignments verified")
