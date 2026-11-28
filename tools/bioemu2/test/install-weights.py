"""Install the pinned weights/cache files from weights.json inside the image."""
import gzip
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import urllib.request

from bioemu2.data_interface import System
from bioemu2.msa.colabfold import cached_colabfold_msa

entries = json.loads(pathlib.Path(sys.argv[1]).read_text())
msa_entries = [e for e in entries if e["url"].startswith("https://api.colabfold.com/")]
for entry in entries:
    if entry in msa_entries:
        continue
    dest = pathlib.Path(entry["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(entry["url"], dest)
    got = hashlib.sha256(dest.read_bytes()).hexdigest()
    assert got == entry["sha256"], f"{dest}: {got} != {entry['sha256']}"
    assert dest.stat().st_size == entry["bytes"], dest
    print(f"{dest} {dest.stat().st_size}")

# ColabFold responses have no permanent per-response URL. Resolve the same
# pinned systems through upstream's build-time client, then verify the exact
# cache keys, contents and checksums recorded by the setup run. The installed
# image never performs these queries: its test and documented examples are
# offline. The AKE example and 1AKE assembly share one sequence and cache key.
systems = []
for entry in entries:
    if "/rcsb/" in entry["dest"]:
        with tempfile.NamedTemporaryFile(suffix=".cif") as local:
            local.write(gzip.decompress(pathlib.Path(entry["dest"]).read_bytes()))
            local.flush()
            systems.append(System.from_cif(local.name))
for name in ("ake.yaml", "chignolin.yaml"):
    systems.append(System.from_yaml(pathlib.Path("/opt/bioemu2/examples") / name))
for system in systems:
    unique_sequences = list(dict.fromkeys(
        entity.sequence for entity in system.entities
        if entity.model_dump(mode="json").get("entity_type") == "protein"))
    print(f"Resolving build-time ColabFold MSA for {[len(s) for s in unique_sequences]} residues",
          flush=True)
    cached_colabfold_msa(unique_sequences, use_pairing=False, use_env=True)
    if len(unique_sequences) > 1:
        cached_colabfold_msa(unique_sequences, use_pairing=True, use_env=False)
cache = pathlib.Path(os.environ["BIOEMU2_MSA_CACHE"])
assert {p.name for p in cache.glob("*.json")} == {pathlib.Path(e["dest"]).name
                                                  for e in msa_entries}
for entry in msa_entries:
    dest = pathlib.Path(entry["dest"])
    assert dest.is_file(), f"Missing expected cached MSA {dest}"
    got = hashlib.sha256(dest.read_bytes()).hexdigest()
    assert got == entry["sha256"], f"{dest}: {got} != {entry['sha256']}"
    assert dest.stat().st_size == entry["bytes"], dest
    print(f"{dest} {dest.stat().st_size}")

print(f"{len(entries)} pinned files installed and verified")
