---
name: hyplm-flash
description: "HyPLM-Flash (Tencent Hunyuan's 650M protein language model) contact maps from the categorical Jacobian, computed exactly as oplm computes them, on its fused bf16 GPU engine. Reach for it to turn protein sequences into long-range contact score maps comparable with OPLM's precision@L curves."
metadata:
  version: 1.0.0
---
# hyplm-flash

Image: `ghcr.io/brineylab/toolshed/hyplm-flash:1.0.0`

HyPLM-Flash, the 650M-parameter masked protein language model trained by Tencent's Hyra research agent (`Tencent-Hunyuan/Hyra-results`, `AI4Science/hyplm_flash` at commit `b24e7f5`: 32 blocks, hidden size 1280, ESM-2's 33-token alphabet), run on its own fused GPU inference engine — padding-free packed batches, Triton kernels, FlashAttention-2 — with `precision="bf16"`. The weights are in the image; it needs no network.

## Commands

```bash
hyplm-flash contacts --in seqs.fasta --out contacts.npz
```

Takes `--device cuda` (the default; the engine has no CPU path, and `--device cpu` is refused).

## Inputs and outputs

- **In:** FASTA, one or more protein sequences in the 20 canonical amino acids, up to 818 residues each (the engine itself takes 8,190; the contacts command stops at 818 like the shed's other PLM tools). The first word of each header is the sequence's id. A sequence that is longer, or holds a non-canonical residue, is refused, naming it.
- **Out:** an `.npz` with one `(L, L)` float32 array per id, `L` being that sequence's length: the residue-only contact score map, with no rows or columns for `<cls>`/`<eos>`. Higher score means a more likely contact. The map is oplm's categorical-Jacobian contact score, the same computation as `oplm`'s structure eval (`compute_categorical_jacobian` + `categorical_jacobian_to_contact_map` at commit 704e01f, copied verbatim into the wrapper): every position is substituted with every one of the 20 amino acids in oplm's `CANONICAL_AMINO_ACIDS` order (substituted, never masked) to build the `(L, 20, L, 20)` Jacobian of bf16-engine logit deltas cast to fp32, which is then centered over all four axes, symmetrized, reduced by the square root of the sum of squares over the two amino-acid axes, diagonal-zeroed, APC-corrected, and symmetrized. The engine's `precision="bf16"` mode runs its projections in BF16; INT8 (the engine's default) is never used.

## Compute

One CUDA GPU, any of the cluster's, with NVIDIA driver 580 or newer (the image is CUDA 13); about 2 GB of GPU memory. The Jacobian costs one engine pass per position per amino acid, in batches of 32 mutants that the engine packs into one padding-free forward pass, so its cost grows with the square of the length: on an A100, 7 s for a 76-residue sequence — the first of a run, whose pass also loads the engine and warms its kernels — and 3 s at 129 residues, about 2 min at 818, extrapolated on that square law; the fixture pair, both sequences end to end through the wrapper with the load, takes about 9 s. There is no CPU path: without a visible GPU the command refuses rather than fall back.

## Running it

On a laptop, from the directory holding `seqs.fasta`:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/hyplm-flash:1.0.0 hyplm-flash contacts --in seqs.fasta --out contacts.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/hyplm-flash:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work hyplm-flash contacts --in seqs.fasta --out contacts.npz
```

## Gotchas

- The code is MIT-licensed and the weights (a 2.6 GB GitHub Release, sha256-verified by the image's `download.sh`) are published under it; the license ships beside the weights at `/opt/model/LICENSE`. Cite the HyPLM technical report (in upstream's `AI4Science/hyplm_flash/`) when you use them.
- Upstream ships only the fused engine, no transformers checkpoint: the wrapper drives `HyPLMEncoder` with `precision="bf16"` (the engine's INT8 default is for embeddings and zero-shot scoring, and is never used here), decoding each batch of mutant token ids back to the protein strings the engine takes.
- The engine needs FlashAttention: the image carries FlashAttention 2.8.3's prebuilt CUDA-13/torch-2.10 wheel, which is why the base image is torch 2.10 rather than the torch 2.14 the shed's other PLM tools use (2.8.3 is the newest release with prebuilt CUDA-13 wheels: the later v2.8.3.post2 maintenance release ships none for CUDA 13 and changes no kernels).
- The Jacobian runs in batches of 32 mutants, and the test's fixture (ubiquitin, 76 residues; lysozyme, 129) crosses that threshold many times over, 129 mutation batches for the pair (48 and 81). Each map is checked against `test/ref.npz`, a reference computed with oplm's own functions at commit 704e01f on the same weights through the engine's bf16 mode: exact symmetry, Pearson r >= 0.98, and top-L long-range (|i - j| >= 24) overlap >= 0.8, the tolerance for GPU kernel noise.
- A large FASTA is one full Jacobian per sequence, quadratic in length: hours past a few hundred sequences, so submit it as a job, not an interactive `srun`.
