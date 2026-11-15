---
name: esm2-650m
description: "ESM-2 650M protein language model: pseudo-log-likelihoods, per-sequence embeddings, masked-position infilling, and categorical-Jacobian contact maps for protein sequences. Reach for it for a sequence score, an embedding, or residue-residue contacts when a structure is not needed, and for sequences up to 1022 residues."
metadata:
  version: 1.1.0
---
# esm2-650m

Image: `ghcr.io/brineylab/toolshed/esm2-650m:1.1.0`

ESM-2 with 650M parameters (`facebook/esm2_t33_650M_UR50D`, 33 layers, 1280-dimensional), run through `transformers`. The weights are in the image; it needs no network. Version 1.1.0 adds `contacts` to 1.0.1's commands, on the same weights: it builds from the published 1.0.1 image by digest.

## Commands

```bash
esm2 pll --in seqs.fasta --out pll.tsv
esm2 embed --in seqs.fasta --out emb.npz
esm2 fill --in masked.fasta --out filled.fasta
esm2 contacts --in seqs.fasta --out contacts.npz
```

Each takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** FASTA, one or more protein sequences in the 20 standard residues. The first word of each header is the sequence's id. For `fill`, write `<mask>` at each position to infill (`MKTAY<mask>AKQR`).
- **`pll`:** a TSV with a header line and columns `id`, `pll`. The pseudo-log-likelihood: each position masked in turn and the natural log-probability of the true residue there, summed over the sequence. It is negative; closer to 0 is more likely under the model. It grows in magnitude with length, so divide by length to compare sequences of different lengths.
- **`embed`:** an `.npz` with `ids` (strings, in input order) and `embeddings` (float32, shape `(n, 1280)`): the last layer, mean-pooled over residues (`<cls>` and `<eos>` excluded). Read it with `numpy.load`.
- **`fill`:** FASTA with the same ids, each `<mask>` replaced by the most likely of the 20 standard residues, all masks predicted in one pass.
- **`contacts`:** an `.npz` holding one `(L, L)` float32 array per id, the contact score map `oplm` computes from the categorical Jacobian (oplm commit `704e01f`, `src/oplm/eval/metrics/categorical_jacobian.py`): every residue substituted with each of the 20 canonical amino acids in oplm's order (no masking, the wild type included), the logit deltas from the unmasked sequence form an `(L, 20, L, 20)` tensor, which is centered over all four axes, symmetrized, reduced to `(L, L)` by the square root of the sum of squares over the two amino-acid axes, zeroed on the diagonal, APC-corrected, and symmetrized. The forward pass runs in bf16; the Jacobian and the reduction are fp32. `contacts` takes only the 20 canonical residues, and the model takes at most 1022 of them — plm-evals' longest structure eval set is 818.

## Compute

One GPU, any of the cluster's, with NVIDIA driver 580 or newer (the image is CUDA 13); about 7 GB of GPU memory at the longest sequence. `pll` costs one forward pass per residue, so it is the slow command at length: on an L40S, about 3 s per 250-residue sequence (some 1,000 an hour) and about a minute at 1022 residues. `embed` and `fill` are one pass per sequence, fast at any length. `contacts` costs 20 forward passes per residue plus one, 20 substituted residues at a time, so its cost grows with the square of the length: about 2 s on a 76-residue sequence (1.4 GB of GPU memory) and about 4 min at 818 residues, extrapolated on that square law, where the fp32 Jacobian alone is 1.1 GB. Loading the model adds about 5 s to each command. A few short sequences are fine on CPU.

## Running it

On a laptop, from the directory holding `seqs.fasta` (drop `--device` without a GPU):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/esm2-650m:1.1.0 esm2 pll --in seqs.fasta --out pll.tsv
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/esm2-650m:1.1.0 \
  --container-mounts="$PWD":/work --container-workdir=/work esm2 contacts --in seqs.fasta --out contacts.npz
```

## Gotchas

- The mask token is `<mask>`, lowercase as written; other residue letters are case-insensitive.
- A sequence longer than 1022 residues is refused, naming it; the model has 1024 positions. `contacts` refuses a sequence holding anything but the 20 standard residues, naming them.
- With an NVIDIA driver older than 580 the GPU is not usable and the default device falls back to CPU, many times slower; `--device cuda` then fails rather than fall back.
- Sequences run one at a time: a large FASTA through `pll` or `contacts` is one or 20 passes per residue, so submit it as a job, not an interactive `srun`.
- `fill` predicts all masks at once, each without seeing the others' predictions.
- `contacts`'s test compares its map to a reference computed once with oplm's own `compute_categorical_jacobian` and `categorical_jacobian_to_contact_map` at commit `704e01f` on the same weights in bf16 (`test/contacts-reference.npz`, beside the fixture `test/contacts.fasta`, whose 76-residue ubiquitin crosses the mutation-batch threshold many times over: 1,520 substitutions at 20 per forward pass). The check is exact symmetry plus a tolerance for GPU kernel noise: Pearson r >= 0.98 over the map and >= 0.8 top-L long-range overlap (pairs with |i - j| >= 24, oplm's long-range separation; measured 0.95 on this model).
- The weights are MIT-licensed: the `facebook/esm2_t33_650M_UR50D` model card declares `license: mit`, and Meta's ESM code repository (facebookresearch/esm) is MIT. The upstream `LICENSE` is shipped beside the weights at `/opt/model/LICENSE`, committed in the tool tree.
