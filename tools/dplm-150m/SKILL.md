---
name: dplm-150m
description: "DPLM v1 150M protein language model (airkingbd/dplm_150m): residue-residue contact maps computed exactly as oplm does, by the categorical-Jacobian pipeline. Reach for it when plm-evals needs a contact map for DPLM."
metadata:
  version: 1.0.5
---
# dplm-150m

Image: `ghcr.io/brineylab/toolshed/dplm-150m:1.0.5`

DPLM v1, the sequence model (30 layers, 640-dimensional, about 150M parameters), from `airkingbd/dplm_150m`, loaded in bf16 through the modeling code vendored from bytedance/dplm. One command, `contacts`, which turns sequences into contact maps exactly the way oplm computes them. The weights are in the image, stored exactly as released (the 595 MB fp32 checkpoint) and cast to bf16 as they load; it needs no network.

## Commands

```bash
dplm contacts --in seqs.fasta --out contacts.npz
```

Takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** FASTA, protein sequences in the 20 canonical amino acids; anything else is refused, naming the sequence. The first word of each header is the id. Up to 1024 residues (the model's positions, less `<cls>` and `<eos>`); the eval sets stay within 818.
- **Out:** an `.npz` holding one `(L, L)` float32 array per id, `L` that sequence's length: residue-residue contact scores, residues only, no rows or columns for special tokens. Read it with `numpy.load`; higher means more contact-like.

**The computation** is oplm's, from briney/oplm at commit `704e01f5ed4b8322bb82e0c961b2c16361c47fa2`, both functions vendored verbatim under `src/oplm/`: the weights and every forward pass run in bf16, the logits are cast to fp32, and the categorical Jacobian `J[i, a, j, b]` — the change in position `j`'s logit for amino acid `b` when position `i` is substituted to amino acid `a` — is built over all positions with both amino-acid axes in oplm's `CANONICAL_AMINO_ACIDS` order, mapped to DPLM's own token ids (they are DPLM vocab ids 4-23, in that order). Substituting the wild-type residue gives a zero row, which is kept; nothing is masked; 20 mutants go in one forward pass. The reduction is `categorical_jacobian_to_contact_map`'s, in order: mean-center over all four axes, symmetrize the 4-D tensor, square root of the sum of squares over the two amino-acid axes, zero the diagonal, average-product correction, symmetrize. Never INT8 or any other quantization.

## Compute

One GPU, any of the cluster's, with NVIDIA driver 580 or newer (the image is CUDA 13), and about 3 GB of GPU memory. The Jacobian costs `L x 20` mutants in batches of 20, plus one wild-type pass, each forward pass of batch 20: on an A100 about 30 s for the 76-residue test fixture and about 5 min at 818 residues. Loading the model adds about 5 s. A few short sequences are fine on CPU, many times slower.

## Running it

On a laptop, from the directory holding `seqs.fasta` (drop `--device` without a GPU):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/dplm-150m:1.0.5 dplm contacts --in seqs.fasta --out contacts.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/dplm-150m:1.0.5 \
  --container-mounts="$PWD":/work --container-workdir=/work dplm contacts --in seqs.fasta --out contacts.npz
```

## Gotchas

- **Licenses.** The weights (`airkingbd/dplm_150m`) carry no license of their own; this tool publishes them under CONTRIBUTING's rule 2, on the official code repository's license: github.com/bytedance/dplm is Apache-2.0, and its license is copied into the image at `/opt/model/LICENSE`. The vendored modeling code (`/opt/src/dplm_modeling_esm.py`) is from that repository, commit `8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d`, with its byprot registration lines removed, its transformers import made explicit, and one bf16 fix: the rotary rotation's fp32 cos/sin promote query and key to fp32 while the value stays bf16, a mix `scaled_dot_product_attention` refuses, so the rotated query and key are cast back to bf16 (the rotation itself stays fp32; the forward runs in bf16, the precision the weights are loaded in).
- **The image ships the released fp32 checkpoint** (595 MB, `pytorch_model.bin`, sha256-pinned in the build); the wrapper casts it to bf16 at load with `torch_dtype=bfloat16` — the same cast, on the same bytes, that the test's reference map was computed with, so the loaded weights are bit-identical to the reference's. Never re-quantize further; this is the model's full bf16 precision, not INT8 or any other quantization.
- **The environment is vendored, not pip-installed.** The pinned wheels under `wheels/` (transformers 4.51.3, tokenizers 0.21.1, huggingface_hub 0.30.2, safetensors 0.5.3, numpy 2.5.4 and their transitives, a closed set for the base image's Python 3.12, sha256-summed) are unpacked at build time into `/opt/vendor` with python's own zipfile: the build never runs pip — it died on the build host mid-install twice, once from the package index and once from these same wheels offline. `dplm` and `test/check.py` put `/opt/vendor` at the front of `sys.path`, ahead of the base image's own site-packages, which contribute only torch and torch's dependencies.
- **Test tolerance.** `test/run.sh` compares the fixture's map against a reference computed once, in a campaign run, with oplm's own `compute_categorical_jacobian` and `categorical_jacobian_to_contact_map` at commit `704e01f5ed4b8322bb82e0c961b2c16361c47fa2` on the same weights in bf16 (committed as `test/reference.npz`, float16). It requires exact symmetry, Pearson r >= 0.98, and top-L long-range overlap >= 0.8 (long-range: `|i - j| >= 24`, oplm's separation), allowing GPU kernel noise.
- **The fixture crosses the batching threshold**: 76 residues x 20 amino acids = 1520 mutants = 76 forward passes, so the mutation-batch loop runs many times, not once.
- A sequence longer than 1024 residues is refused, naming it; the model has 1026 positions. The eval sets stay within 818.
- With an NVIDIA driver older than 580 the GPU is not usable and the default device falls back to CPU, many times slower; `--device cuda` then fails rather than fall back.
- Sequences run one at a time: a large FASTA through `contacts` is many minutes per sequence at full length, so submit it as a job, not an interactive `srun`.
