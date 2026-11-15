---
name: proteinbert
description: "ProteinBERT (nadavbra/protein_bert, the released full-GO pretrained dump) contact maps from the categorical Jacobian, computed exactly as oplm computes them. Reach for it to turn protein sequences into long-range contact score maps comparable with OPLM's precision@L curves."
metadata:
  version: 1.0.0
---
# proteinbert

Image: `ghcr.io/brineylab/toolshed/proteinbert:1.0.0`

ProteinBERT, the 16M-parameter model of Ofer & Brandes (Bioinformatics 2022): 6 blocks of per-residue convolutions beside a global-attention track, hidden 128 per residue and 512 global, 4 heads, trained on ~106M sequences with their GO annotations. Its own code is vendored (TensorFlow/Keras 2, from github.com/nadavbra/protein_bert at commit `69a1122`); the weights are the released full-GO pretrained dump (`epoch_92400_sample_23500000.pkl`, 8,943 annotations), in the image. It needs no network.

## Commands

```bash
proteinbert contacts --in seqs.fasta --out contacts.npz
```

Takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** FASTA, one or more protein sequences in the 20 canonical amino acids, up to 818 residues each. The first word of each header is the sequence's id. A sequence that is longer, or holds a non-canonical residue, is refused, naming it.
- **Out:** an `.npz` with one `(L, L)` float32 array per id, `L` being that sequence's length: the residue-only contact score map, with no rows or columns for `<START>`/`<END>`. Higher score means a more likely contact. The map is oplm's categorical-Jacobian contact score, the same computation as `oplm`'s structure eval (`compute_categorical_jacobian` + `categorical_jacobian_to_contact_map` at commit 704e01f, ported to numpy in the wrapper, operation for operation): the weights load and the forward passes run in bf16 (Keras global policy; never INT8 or any other quantization), logits are cast to fp32 inside the graph, every position is substituted with every one of the 20 amino acids in oplm's `CANONICAL_AMINO_ACIDS` order to build the `(L, 20, L, 20)` Jacobian of logit deltas, which is then centered over all four axes, symmetrized, reduced by the square root of the sum of squares over the two amino-acid axes, diagonal-zeroed, APC-corrected, and symmetrized.

## Compute

One GPU, any of the cluster's (the image carries its own CUDA 12.2 stack, so any NVIDIA driver 525 or newer works); the model is small, so a few hundred MB of GPU memory at the longest sequence and about 1 GB of host RAM for the Jacobian at 818 residues. The Jacobian costs one forward pass per position per amino acid, in batches of 64 mutants, so its cost grows with the square of the length; the model is also rebuilt at exactly `L + 2` tokens per sequence (its weights are length-independent but its convolutions and global attention are not padding-invariant), which adds a second or two per sequence. Measured on an A100: 5.5 s for a 76-residue sequence and 3.8 s at 129 residues (the first of a run pays TensorFlow's graph tracing and CUDA warmup), and about 2-3 min at 818, extrapolated from that square law; the whole fixture run — Python and TensorFlow startup, the weights load and verification, both sequences — takes about 13 s. A few short sequences run on CPU too, slower.

## Running it

On a laptop, from the directory holding `seqs.fasta` (drop `--device` without a GPU):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/proteinbert:1.0.0 proteinbert contacts --in seqs.fasta --out contacts.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/proteinbert:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work proteinbert contacts --in seqs.fasta --out contacts.npz
```

## Gotchas

- The weights are CC-BY-4.0 (the authors' Zenodo archive of the same snapshot carries the grant), shipped beside the weights at `/opt/model/LICENSE` with provenance in `/opt/model/NOTICE`; the vendored code is MIT (declared by upstream's setup.py, which carries no license file; see the tool's `LICENSE`). Credit Dan Ofer and Nadav Brandes and cite their paper when you use them.
- The image ships the released dump (192 MB) as one file, fetched by a single short build step and sha256-pinned; the wrapper verifies the same sha256 again before it loads it, so the weights are pinned twice.
- This is the full-GO pretrained dump (`epoch_92400_sample_23500000.pkl`, `n_annotations=8943`), the one ProteinBERT's README names; the annotation input the wrapper feeds is all zeros, as upstream's `InputEncoder` does — the GO head is loaded but not used.
- The vendored model code is Keras-2-era, which is why the image pins TensorFlow 2.15.1 (the last release whose `keras` is that), whose wheels stop at Python 3.11.
- The one vendored change, marked in place: the `output-seq` head is built with `activation=None`, so the model returns its pre-softmax logits. In oplm's categorical Jacobian the wild-type term cancels under the four-axis mean-centering, so log-probabilities and logits give the same map — but in bf16, log-softmax would underflow where the logits do not. The weights and everything else are upstream's, untouched.
- Residue `i` is token `i + 1` between `<START>` and `<END>`; a sequence that does not tokenize that way is refused.
- The Jacobian runs in batches of 64 mutants, and the test's fixture (ubiquitin, 76 residues; lysozyme, 129) crosses that threshold many times over, 24 mutation batches for the 76-residue sequence and 41 for the 129-residue one. Each map is checked against `test/ref.npz`, a reference computed with oplm's own functions at commit 704e01f on the same bf16 weights (the logits coming from the vendored model, pre-softmax): exact symmetry, Pearson r >= 0.98, and top-L long-range (|i - j| >= 24) overlap >= 0.8, the tolerance for GPU kernel noise.
- With an NVIDIA driver older than 525 the GPU is not usable and the default device falls back to CPU, slower; `--device cuda` then fails rather than fall back.
- A large FASTA is one full Jacobian per sequence, quadratic in length: hours past a few hundred sequences, so submit it as a job, not an interactive `srun`.
