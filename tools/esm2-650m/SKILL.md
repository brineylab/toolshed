---
name: esm2-650m
description: "ESM-2 650M protein language model: pseudo-log-likelihoods, per-sequence embeddings, and masked-position infilling for protein sequences. Reach for it for a sequence score or an embedding when a structure is not needed."
metadata:
  version: 1.0.0
---
# esm2-650m

Image: `ghcr.io/brineylab/toolshed/esm2-650m:1.0.0`

ESM-2 with 650M parameters (`facebook/esm2_t33_650M_UR50D`, 33 layers, 1280-dimensional), run through `transformers`. The weights are in the image; it needs no network.

## Commands

```bash
esm2 pll --in seqs.fasta --out pll.tsv
esm2 embed --in seqs.fasta --out emb.npz
esm2 fill --in masked.fasta --out filled.fasta
```

Each takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** FASTA, one or more protein sequences in the 20 standard residues. The first word of each header is the sequence's id. For `fill`, write `<mask>` at each position to infill (`MKTAY<mask>AKQR`).
- **`pll`:** a TSV with a header line and columns `id`, `pll`. The pseudo-log-likelihood: each position masked in turn and the natural log-probability of the true residue there, summed over the sequence. It is negative; closer to 0 is more likely under the model. It grows in magnitude with length, so divide by length to compare sequences of different lengths.
- **`embed`:** an `.npz` with `ids` (strings, in input order) and `embeddings` (float32, shape `(n, 1280)`): the last layer, mean-pooled over residues (`<cls>` and `<eos>` excluded). Read it with `numpy.load`.
- **`fill`:** FASTA with the same ids, each `<mask>` replaced by the most likely of the 20 standard residues, all masks predicted in one pass.

## Compute

One GPU, any of the cluster's; MEM of GPU memory. `pll` costs one forward pass per residue, so it is the slow command: RATE. `embed` and `fill` are one pass per sequence, fast at any length. A few short sequences are fine on CPU.

## Running it

On a laptop, from the directory holding `seqs.fasta` (drop `--device` without a GPU):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/esm2-650m:1.0.0 esm2 pll --in seqs.fasta --out pll.tsv
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/esm2-650m:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work esm2 pll --in seqs.fasta --out pll.tsv
```

## Gotchas

- The mask token is `<mask>`, lowercase as written; other residue letters are case-insensitive.
- A sequence longer than 1022 residues is refused, naming it; the model has 1024 positions. Split it or crop it.
- Sequences run one at a time: a large FASTA through `pll` is hours, so submit it as a job, not an interactive `srun`.
- `fill` predicts all masks at once, each without seeing the others' predictions.
