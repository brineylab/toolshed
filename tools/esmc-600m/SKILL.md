---
name: esmc-600m
description: "ESM C 600M protein language model: pseudo-log-likelihoods, per-sequence embeddings, masked-position infilling, and categorical-Jacobian contact maps for protein sequences. Reach for it for a sequence score, an embedding, or residue-residue contacts when a structure is not needed, and for sequences up to 2046 residues."
metadata:
  version: 1.1.0
---
# esmc-600m

Image: `ghcr.io/brineylab/toolshed/esmc-600m:1.1.0`

ESM C with 600M parameters (`biohub/ESMC-600M`, 36 layers, 1152-dimensional), run through the ESM C port in `transformers`. The weights are in the image; it needs no network. Its commands and files are those of `esm2-650m`, so the two compare directly. 1.1.0 is 1.0.1 plus `contacts`: the same pinned weights (revision `fcc9d36`, fp32) and the same environment, built `FROM` the published 1.0.1 image by digest, so `pll`, `embed`, and `fill` give 1.0.1's numbers unchanged.

## Commands

```bash
esmc pll --in seqs.fasta --out pll.tsv
esmc embed --in seqs.fasta --out emb.npz
esmc fill --in masked.fasta --out filled.fasta
esmc contacts --in seqs.fasta --out contacts.npz
```

Each takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** FASTA, one or more protein sequences in the 20 standard residues. The first word of each header is the sequence's id. For `fill`, write `<mask>` (or `_`, ESM C's short form) at each position to infill (`MKTAY<mask>AKQR`).
- **`pll`:** a TSV with a header line and columns `id`, `pll`. The pseudo-log-likelihood: each position masked in turn and the natural log-probability of the true residue there, summed over the sequence. It is negative; closer to 0 is more likely under the model. It grows in magnitude with length, so divide by length to compare sequences of different lengths.
- **`embed`:** an `.npz` with `ids` (strings, in input order) and `embeddings` (float32, shape `(n, 1152)`): the last layer, mean-pooled over residues (`<cls>` and `<eos>` excluded). Read it with `numpy.load`.
- **`fill`:** FASTA with the same ids, each mask replaced by the most likely of the 20 standard residues, all masks predicted in one pass.
- **`contacts`:** an `.npz` holding one `(L, L)` float32 array per id, the contact score map `oplm` computes from the categorical Jacobian (oplm commit `704e01f`, `src/oplm/eval/metrics/categorical_jacobian.py`): every residue substituted with each of the 20 canonical amino acids in oplm's order (no masking, the wild type included), the logit deltas from the unmasked sequence form an `(L, 20, L, 20)` tensor, which is centered over all four axes, symmetrized, reduced to `(L, L)` by the square root of the sum of squares over the two amino-acid axes, zeroed on the diagonal, APC-corrected, and symmetrized. The forward pass runs in bf16; the Jacobian and the reduction are fp32. `contacts` takes sequences of at most 818 residues (plm-evals' longest structure eval set) holding only the 20 canonical residues; others are refused, naming them.

## Compute

One GPU, any of the cluster's, with NVIDIA driver 580 or newer (the image is CUDA 13); about 8 GB of GPU memory at the longest sequence. `pll` costs one forward pass per residue, so it is the slow command: on an L40S, about 3 s per 250-residue sequence (some 1,000 an hour), about a minute at 1022 residues, and about 5 minutes at 2046. Loading the model adds about 5 s to each command. `embed` and `fill` are one pass per sequence, fast at any length. `contacts` costs 20 forward passes per residue plus one, 64 substituted residues at a time: on an A100, 10.7 s for ubiquitin and lysozyme together (205 residues, 4,100 substitutions, the model's 5 s load included), about 3 s per 100 residues at these lengths and minutes at 818 residues, where its fp32 Jacobian alone is 1.1 GB. A few short sequences are fine on CPU.

## Running it

On a laptop, from the directory holding `seqs.fasta` (drop `--device` without a GPU):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/esmc-600m:1.1.0 esmc pll --in seqs.fasta --out pll.tsv
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/esmc-600m:1.1.0 \
  --container-mounts="$PWD":/work --container-workdir=/work esmc contacts --in seqs.fasta --out contacts.npz
```

## Gotchas

- `_` in a sequence is a mask, not a gap.
- A sequence longer than 2046 residues is refused, naming it; the model has 2048 positions. `contacts` refuses earlier, at 818 residues.
- With an NVIDIA driver older than 580 the GPU is not usable and the default device falls back to CPU, many times slower; `--device cuda` then fails rather than fall back.
- Sequences run one at a time: a large FASTA through `pll` or `contacts` is hours, so submit it as a job, not an interactive `srun`.
- `fill` predicts all masks at once, each without seeing the others' predictions.
- `contacts`'s test compares its maps to a reference computed once with oplm's own `compute_categorical_jacobian` and `categorical_jacobian_to_contact_map` at commit `704e01f` on the same weights in bf16 (`test/contacts-reference.npz`, beside the fixture `test/contacts.fasta`, whose two sequences cross the mutation-batch threshold many times over: 1,520 and 2,580 substitutions at 64 per forward pass). The check is exact symmetry plus a tolerance for GPU kernel noise: Pearson r >= 0.98 over the map and >= 0.8 top-L long-range overlap (pairs with |i - j| >= 24, oplm's long-range separation).
- The weights' license is the one on the `biohub/ESMC-600M` model card (MIT when this image was built); earlier ESM C releases under EvolutionaryScale were non-commercial, so check which one a citation names. The image carries Biohub's `LICENSE.md` and `THIRD_PARTY_NOTICE.md` beside the weights.
