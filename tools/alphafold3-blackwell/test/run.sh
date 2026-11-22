#!/usr/bin/env bash
# The alphafold3-blackwell tool test: the environment, the CLI, the library
# path, and the input-dialect contract on a small fixture.
#
# What it can and cannot cover: the image deliberately carries NO AlphaFold 3
# parameters (Google DeepMind's WEIGHTS_TERMS_OF_USE.md forbids
# redistribution), so no folding happens here. The test proves the python is
# the AF3 venv's, jax drives the GPU, run_alphafold.py answers --help,
# make_model_config/ModelRunner import and construct, and the input contract
# holds: 'alphafold3' dialect, version 4, "protein" sequence keys, required
# modelSeeds and per-chain ids, unpairedMsaPath read, and the templates
# semantics ("templates": [] = off, absent = search). The first real folding
# check is the caller's smoke run with mounted parameters.
set -eu
cd /tmp

echo "== python =="
python3 --version
python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version"

echo "== jax on the GPU =="
python3 -I - <<'PY'
import jax

devs = jax.devices()
gpus = [d for d in devs if d.platform == "gpu"]
assert gpus, f"no GPU device among {devs}"
d = gpus[0]
print("jax", jax.__version__)
print("device", d.device_kind, "compute capability", d.compute_capability)
PY

echo "== alphafold3 package =="
python3 -I -c "import alphafold3; print('alphafold3 at', alphafold3.__file__)"

echo "== run_alphafold.py --help =="
python3 /app/alphafold/run_alphafold.py --help > ra_help.txt 2> ra_help.err
grep -q "model_dir" ra_help.txt
echo "run_alphafold.py answers --help and documents --model_dir"

echo "== the library path (make_model_config + ModelRunner) =="
python3 -I - <<'PY'
import importlib.util
import pathlib

import jax

spec = importlib.util.spec_from_file_location("ra", "/app/alphafold/run_alphafold.py")
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)
cfg = ra.make_model_config(num_diffusion_samples=5, num_recycles=10)
assert cfg.heads.diffusion.eval.num_samples == 5
assert cfg.num_recycles == 10
runner = ra.ModelRunner(cfg, jax.devices()[0], pathlib.Path("/tmp/models"))
print("ModelRunner", type(runner).__name__, "(parameters NOT loaded in this test)")
PY

echo "== input dialect contract (fixture) =="
python3 -I - <<'PY'
from alphafold3.common import folding_input

with open("/toolshed/test/fold_input.json") as f:
    fi = folding_input.Input.from_json(f.read())
assert fi.rng_seeds == (0,), fi.rng_seeds
assert [c.id for c in fi.chains] == ["A", "B"], [c.id for c in fi.chains]
for c in fi.chains:
    assert c.templates == (), f"templates not off: {c.templates!r}"
assert "EVQL" in fi.chains[0].unpaired_msa, "chain A MSA not read"
assert "DVVMT" in fi.chains[1].unpaired_msa, "chain B MSA not read"
print(fi.name, "chains", [c.id for c in fi.chains],
      "templates", [c.templates for c in fi.chains], "seeds", fi.rng_seeds)

with open("/toolshed/test/fold_input_search.json") as f:
    fs = folding_input.Input.from_json(f.read())
for c in fs.chains:
    assert c.templates is None, f"absent templates must mean search: {c.templates!r}"
print("absent templates key parses as search (None), as documented")
PY

echo "TEST COMPLETE"
