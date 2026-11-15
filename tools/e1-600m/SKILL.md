---
name: e1-600m
description: "Profluent E1-600M protein language model: categorical-Jacobian contact maps for protein sequences, in single-sequence mode only (no retrieved or MSA context). Reach for it for an (L, L) contact map computed exactly as oplm's structure eval computes it, for sequences up to 818 residues."
metadata:
  version: 1.0.0
---
# e1-600m

Image: `ghcr.io/brineylab/toolshed/e1-600m:1.0.0`

Profluent's E1-600M (`Profluent-Bio/E1-600m`, 641M parameters, 30 layers, 1280-dimensional), run through Profluent's own `E1` package (github.com/Profluent-AI/E1 at commit `bfd2620`). The image rides the shed's shared CUDA-13 runtime stack — the same `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime` base esm2-650m and esmc-600m build on — rather than E1's own locked torch 2.8.0 (its pyproject pins torch `>=2.7,<2.9`); the campaign that computed the reference maps ran E1 under both stacks and measured the drift between them, and the Gotchas carry the numbers. The weights are in the image; it needs no network.

## Commands

```bash
e1-600m contacts --in seqs.fasta --out contacts.npz [--device cuda|cpu]
```

`--device` defaults to the GPU when one is visible.

## Inputs and outputs

- **In:** FASTA, one or more protein sequences in the 20 canonical amino acids (others are refused, named; the sequences are uppercased first). The first word of each header is the sequence's id. At most 818 residues per sequence (longer is refused, named): the contacts contract's limit, though E1 itself takes far longer.
- **Out:** an `.npz` with one `(L, L)` float32 array per id, `L` that sequence's length: the residue-level contact score map. Residues only: no rows or columns for special tokens.

**The computation** is the one OPLM's structure eval performs (github.com/briney/oplm at commit `704e01f5`, `src/oplm/eval/metrics/categorical_jacobian.py`; the wrapper copies those functions verbatim rather than pull oplm's training-stack dependencies into the image):

1. The weights load in bf16 and the forward pass runs in bf16; logits, the Jacobian and the reduction are fp32. Never INT8 or any other quantization.
2. One wild-type forward pass; the logits at each residue position, restricted to the 20 canonical amino acids in oplm's `CANONICAL_AMINO_ACIDS` order: an `(L, 20)` array.
3. For each position `i` and each of the 20 amino acids (same order, each mapped to E1's own token ids): substitute (no masking), one forward pass, subtract the wild type's logits. The result is the `(L, 20, L, 20)` categorical Jacobian; the wild-type substitution's zero row is kept. Mutants go through the model 20 at a time.
4. The reduction, in order: center over all four axes; symmetrize the 4-D tensor; square root of the sum of squares over the two amino-acid axes; zero the diagonal; APC; symmetrize the `(L, L)` map.

E1 runs in single-sequence mode only: no retrieved or MSA context, one sequence per input row, tokenized as `<bos> 1 <residues> 2 <eos>` by E1's own batch preparer.

## Compute

One GPU, any of the cluster's, with NVIDIA driver 580 or newer (the image is CUDA 13); about 5 GB of GPU memory at the longest sequence. `contacts` costs one forward pass per 20 mutants — an `L`-residue sequence takes `L` batched passes plus the wild type — so a large FASTA is hours: submit it as a job, not an interactive `srun`. On an A100, a 129-residue sequence takes about 15 seconds once the model is loaded, and model load plus the one-time attention-kernel compilation add about a minute to the first sequence; at 818 residues, about 4 hours per sequence. A few short sequences are fine on CPU, many times slower.

## Running it

On a laptop, from the directory holding `seqs.fasta` (drop `--device` without a GPU):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/e1-600m:1.0.0 e1-600m contacts --in seqs.fasta --out contacts.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/e1-600m:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work e1-600m contacts --in seqs.fasta --out contacts.npz
```

## Gotchas

- The image runs E1 on torch 2.14.1 — the CUDA-13 runtime base the shed's other GPU tools share — not the torch 2.8.0 E1's own pixi.lock pins (its pyproject asks for `>=2.7,<2.9`); riding that base is what keeps the image a fraction of the size of a from-slim torch install. The campaign that computed the committed reference maps also computed them under E1's locked torch 2.8.0 stack and compared the two: the torch 2.14.1 maps agree with the locked-stack ones at Pearson r 0.9995 (lysozyme, 129 residues) and 0.9983 (ubiquitin, 76 residues), top-L long-range overlap 0.984 and 0.947 — inside the test's own tolerance (Pearson r ≥ 0.98, top-L long-range overlap ≥ 0.8). The committed reference maps are the torch 2.14.1 ones, computed on the image's exact stack.
- The weights are Profluent Bio Inc.'s E1-600m, licensed under the Profluent-E1 Clickthrough License Agreement, which allows redistribution under conditions: the image ships Profluent's `LICENSE`, `NOTICE` and `ATTRIBUTION` beside the weights in `/opt/model`, and the program prints a "Profluent-E1" attribution line to stderr each time it starts, as the license's Attribution Guidelines require. Each license file rides its own `COPY` line in the Containerfile.
- The image's build pulls its weights over three routes in turn — plain HTTPS from huggingface.co's file CDN with hf-xet disabled, the hf-xet protocol, then curl — each from a clean state, and verifies the pinned sha256 of what lands before the build continues. `fetch-weights.py` in the tool's directory is that step. The tool at run time still needs no network.
- Single-sequence mode only: a comma in a sequence (E1's retrieved-context syntax) is refused with the non-canonical characters, as is any residue outside the 20 canonical amino acids.
- The map is exactly symmetric by construction; the test checks it bitwise.
- E1's modeling module imports `kernels` unconditionally and calls `get_kernel` at import time, which would swap in an unpinned triton layer-norm fetched from the Hugging Face hub whenever the network or a warm cache let it; the wrapper stubs `kernels.get_kernel` to raise, so E1 takes its own PyTorch `rms_norm` fallback everywhere, online or not. The wrapper imports `transformers` before installing the stub, because transformers' import-time availability probe calls `find_spec("kernels")`, which a spec-less stub in `sys.modules` would crash. The committed reference maps were computed on the same path.
- The first `contacts` call on a GPU compiles E1's flex attention through `torch.compile`/Triton, which needs a system compiler at run time: the image carries `gcc` and `g++` (boltz-2 1.0.1's lesson). Compilation artifacts land in the caller's cache directories and are reused across runs.
- The test's fixture (ubiquitin, 76 residues; lysozyme, 129) crosses the mutation-batching threshold: mutants go through the model 20 at a time, so the fixture exercises 76 and 129 batched forward passes per sequence, the path every real input takes.
- The test's tolerance: exact symmetry, plus Pearson r ≥ 0.98 and top-L long-range (|i − j| ≥ 24, oplm's definition) overlap ≥ 0.8 against the committed reference map, which a campaign run computed with oplm's own `compute_categorical_jacobian` and `categorical_jacobian_to_contact_map` on these weights in bf16.
- `--device cuda` without a visible GPU fails, naming the problem; it does not silently fall back.
- With an NVIDIA driver older than 580 the GPU is not usable and the default device falls back to CPU, many times slower.
