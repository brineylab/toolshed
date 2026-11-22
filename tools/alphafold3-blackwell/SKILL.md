---
name: alphafold3-blackwell
description: "Stock AlphaFold 3 v3.0.2 environment (tmessmer's Blackwell build of google-deepmind/alphafold3): jax, the alphafold3 package and run_alphafold.py, ready for structure prediction on Blackwell GPUs. Reach for it when you need AlphaFold 3 predictions with stock, comparable runtimes; the model parameters are NOT in the image and are mounted by the caller."
metadata:
  version: 1.0.0
---
# alphafold3-blackwell

Image: `ghcr.io/brineylab/toolshed/alphafold3-blackwell:1.0.0`

A thin wrapper over the public stock AlphaFold 3 container
`docker.io/tmessmer/alphafold3-blackwell:v3.0.2` (the v3.0.2 build of
`github.com/google-deepmind/alphafold3`'s own `docker/Dockerfile`: Ubuntu
24.04, CUDA 12.6.3, python 3.12, the AF3 venv at `/alphafold3_venv` on `PATH`,
the source at `/app/alphafold`, HMMER 3.4 at `/hmmer/bin`, and the image-level
`XLA_FLAGS`/allocator defaults the upstream Dockerfile sets). Nothing else is
added: no code, no packages, no weights. Use it whenever a run needs stock
AlphaFold 3 — in particular for runtime comparisons, where an accelerated fork
(Nitesh Mishra's setup, AlphaFast) would not give comparable numbers.

## Commands

The image's entrypoint is ignored under pyxis; invoke its python explicitly.
The working directory the image declares is `/app/alphafold`.

Predict from one input JSON, with the data pipeline off (the JSON itself
carries the MSAs and disables templates — see *Inputs and outputs*):

```bash
python3 /app/alphafold/run_alphafold.py \
  --json_path=/path/to/fold_input.json \
  --model_dir=/path/to/weights_dir \
  --output_dir=/path/to/out \
  --run_data_pipeline=false
```

Drive the library in-process instead of the CLI (many targets per model load;
`ModelRunner` and `make_model_config` are `run_alphafold.py`'s own):

```bash
python3 -I -c "import importlib.util, pathlib, jax; spec=importlib.util.spec_from_file_location('ra', '/app/alphafold/run_alphafold.py'); ra=importlib.util.module_from_spec(spec); spec.loader.exec_module(ra); cfg=ra.make_model_config(num_diffusion_samples=5, num_recycles=10); r=ra.ModelRunner(cfg, jax.devices()[0], pathlib.Path('/path/to/weights_dir')); print(type(r))"
```

Check the environment (python, jax devices, package version):

```bash
python3 -I -c "import jax, alphafold3; print(jax.devices()); print(alphafold3.__file__)"
```

## Inputs and outputs

`--model_dir` must contain `af3.bin.zst`, the standard AlphaFold 3
parameters. **The parameters are not, and never will be, in this image**:
Google DeepMind's `WEIGHTS_TERMS_OF_USE.md` forbids redistributing them. The
caller mounts a directory holding `af3.bin.zst` (flyloft mounts the campaign
directory into tool runs) and passes its path as `--model_dir`. Verify the
file's sha256 yourself before use, and never copy it off the cluster it lives
on.

The input JSON must be the `alphafold3` dialect, `version: 4`: a `name`,
`modelSeeds` (required, a list of ints), and `sequences` as a list of
single-key dicts keyed `"protein"` (not the AlphaFold Server dialect's
`"proteinChain"`), each holding at least `id` (required, unique per chain),
`sequence`, and the MSA either inline (`"unpairedMsa"`) or as a path
(`"unpairedMsaPath"`, resolved against the JSON's directory when relative).
Templates per chain: absent means *search* (needs the data pipeline and a PDB
mirror); `"templates": []` means *explicitly off*. With the data pipeline off,
every chain needs its MSA set and templates off:

```json
{"name": "example", "dialect": "alphafold3", "version": 4,
 "modelSeeds": [0], "sequences": [
   {"protein": {"id": "A", "sequence": "EVQLVESGGGLVQPGGSLRLSCAAS",
                "unpairedMsaPath": "/abs/path/A.a3m", "templates": []}},
   {"protein": {"id": "B", "sequence": "DVVMTQTPLTLSVTIGGPASATCKCK",
                "unpairedMsaPath": "/abs/path/B.a3m", "templates": []}}]}
```

Outputs of the CLI: `<output_dir>/<sanitised name>/seed-<s>_sample-<i>/`
mmCIFs and confidence JSONs, a `<name>_ranking_scores.csv`, and the
top-ranked sample copied to the job directory. The ranking metric is
`ranking_score` (ipTM-weighted; AF3's own confidence).

## Compute

One GPU per run (a second is unused: `run_alphafold.py` takes device 0 only).
The upstream Dockerfile sizes the container for up to 5,120 tokens on an
80 GB A100 with the XLA allocator at 95%; a B200 has headroom. A
protein-complex prediction (one seed, 5 diffusion samples, 10 recycles) runs
in minutes on one B200, minutes that scale with token count and MSA depth;
measure your own per-target wall time rather than trusting anyone's average.
CPU featurisation is short; RAM 32G is plenty for inputs of this size.

## Running it

On a laptop (parameters mounted from a local directory):

```bash
podman run --rm --security-opt=label=disable --gpus all \
  -v $HOME/models:/models:ro -v $PWD/in:/in -v $PWD/out:/out \
  ghcr.io/brineylab/toolshed/alphafold3-blackwell:1.0.0 \
  python3 /app/alphafold/run_alphafold.py \
  --json_path=/in/fold_input.json --model_dir=/models \
  --output_dir=/out --run_data_pipeline=false
```

On a Slurm cluster with pyxis (parameters under the campaign directory,
which flyloft mounts into the run):

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/alphafold3-blackwell:1.0.0 \
  python3 /app/alphafold/run_alphafold.py \
  --json_path=$FLYLOFT_CAMPAIGN/data/inputs/target.json \
  --model_dir=$FLYLOFT_CAMPAIGN/weights/af3 \
  --output_dir=$FLYLOFT_OUT/target --run_data_pipeline=false
```

## Gotchas

- **The weights' license.** AlphaFold 3 parameters are under Google
  DeepMind's `WEIGHTS_TERMS_OF_USE.md`: received directly from Google, not
  redistributable. This image ships the code only (upstream's own
  `docker/Dockerfile` likewise ships no parameters; `fetch_weights.sh` in
  the repo is how a user obtains them, and the terms travel with the
  weights). Point `--model_dir` at a mounted copy and leave it where it
  lives.
- **The image test cannot fold.** The test checks the environment (python,
  jax on the GPU, the package, the CLI's `--help`), the input-dialect
  contract, and the templates-off semantics by parsing a fixture; a real
  prediction needs the mounted parameters, so the first true folding check
  is the caller's smoke run. The threshold the fixture does cross is the
  input contract itself: the dialect key (`"protein"`, not
  `"proteinChain"`), `version: 4`, required `modelSeeds` and per-chain `id`,
  and the `templates: []` off-state — the things every caller gets wrong
  first.
- **Stock runtimes only.** For benchmark comparability use this image as
  is; do not swap in an accelerated fork (nmishra's setup, AlphaFast) and
  expect comparable wall times. The image inherits upstream's
  `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` and the 95%-preallocation
  allocator defaults (`XLA_PYTHON_CLIENT_PREALLOCATE=true`,
  `XLA_CLIENT_MEM_FRACTION=0.95`); a caller measuring peak GPU memory must
  set `XLA_PYTHON_CLIENT_PREALLOCATE=false` *before importing jax*, or every
  run reports the preallocated pool.
- **Dialects differ.** `"proteinChain"` JSONs (AlphaFold Server, Protenix
  format) are rejected by `folding_input.Input.from_json`; translate to the
  `"protein"` key, add `dialect: "alphafold3"`, `version: 4`, `modelSeeds`
  and per-chain `id`s first.
- **`--help` exits 1.** absl-py's help flags (`--help`, `--helpshort`,
  `--helpfull`) print the usage to stdout and exit with code 1 by design
  (the gflags convention); a script wrapping the CLI must tolerate that
  nonzero code rather than treat it as a failure. The help text itself is
  the contract: it documents `--json_path`, `--model_dir`,
  `--run_data_pipeline` and friends.
- **pyxis ignores entrypoints**; run `python3` (the venv's, already first on
  `PATH`) with explicit script paths. `/hmmer/bin` is on `PATH` for the data
  pipeline, which is off in the example above.
