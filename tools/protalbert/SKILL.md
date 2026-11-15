---
name: protalbert
description: "ProtALBERT (Rostlab/prot_albert, the UniRef100 model) contact maps from the categorical Jacobian, computed exactly as oplm computes them. Reach for it to turn protein sequences into long-range contact score maps comparable with OPLM's precision@L curves."
metadata:
  version: 1.0.0
---
# protalbert

Image: `ghcr.io/brineylab/toolshed/protalbert:1.0.0`

ProtALBERT, the 224M-parameter ALBERT masked language model trained on UniRef100 (`Rostlab/prot_albert`: 12 layers sharing one group of weights, hidden size 4096, a 34-token SentencePiece vocabulary), run through `transformers`. The weights are in the image; it needs no network.

## Commands

```bash
protalbert contacts --in seqs.fasta --out contacts.npz
```

Takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** FASTA, one or more protein sequences in the 20 canonical amino acids, up to 818 residues each. The first word of each header is the sequence's id. A sequence that is longer, or holds a non-canonical residue, is refused, naming it.
- **Out:** an `.npz` with one `(L, L)` float32 array per id, `L` being that sequence's length: the residue-only contact score map, with no rows or columns for `[CLS]`/`[SEP]`. Higher score means a more likely contact. The map is oplm's categorical-Jacobian contact score, the same computation as `oplm`'s structure eval (`compute_categorical_jacobian` + `categorical_jacobian_to_contact_map` at commit 704e01f, copied verbatim into the wrapper): the weights load and the forward passes run in bf16 (never INT8 or any other quantization), logits are cast to fp32, every position is substituted with every one of the 20 amino acids in oplm's `CANONICAL_AMINO_ACIDS` order to build the `(L, 20, L, 20)` Jacobian of logit deltas, which is then centered over all four axes, symmetrized, reduced by the square root of the sum of squares over the two amino-acid axes, diagonal-zeroed, APC-corrected, and symmetrized.

## Compute

One GPU, any of the cluster's, with NVIDIA driver 580 or newer (the image is CUDA 13); about 8 GB of GPU memory at the longest sequence, hidden size 4096 costing more than a BERT's. The Jacobian costs one forward pass per position per amino acid, in batches of 32 mutants, so its cost grows with the square of the length: on an A100, 4.5 s for a 76-residue sequence and 7.7 s at 129 residues (the first of a run pays CUDA warmup), and about 3-5 min at 818, extrapolated from that square law; loading the model adds about 13 s, and the first run on a machine also assembles and sha256-verifies the checkpoint from its committed chunks, a few seconds of I/O cached beside the system's temp directory thereafter. A few short sequences run on CPU too, many times slower.

## Running it

On a laptop, from the directory holding `seqs.fasta` (drop `--device` without a GPU):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/protalbert:1.0.0 protalbert contacts --in seqs.fasta --out contacts.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/protalbert:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work protalbert contacts --in seqs.fasta --out contacts.npz
```

## Gotchas

- The weights are the ProtTrans pretrained models, which the ProtTrans project releases under the Academic Free License v3.0; the license is shipped beside the weights at `/opt/model/LICENSE`, committed in `model/` from the same revision. Cite Elnaggar et al., ProtTrans (IEEE TPAMI 2021) when you use them.
- The image ships the fp32 `pytorch_model.bin` (0.9 GB) of `Rostlab/prot_albert` @ `2384f557` as seven `128 MiB` chunks, `/opt/model/pytorch_model.bin.part00`–`.part06`, each fetched by one short build step and sha256-pinned (`fetch_chunk.py` carries the manifest); on its first run the wrapper assembles them, checks the whole file's sha256 (`5f616acb…`), and loads it with `dtype=bfloat16` — bit for bit the same bf16 weights a build-time conversion would have shipped, verified on CPU in a campaign run comparing the cast-at-load state dict and a forward pass against the converted file, so no quantization is involved and the reference still matches. Chunked because eight single-step builds of protbert, this shed's first ProtTrans tool, died on the machine that builds this image (connections to huggingface.co that stall, then silent deaths in or just after the long weights step): short steps keep each committed layer small, and since the RUN instructions never change, a failed build's completed chunks stay in the builder's layer cache for the next attempt. The config, the sentencepiece tokenizer and the license are committed in `model/` too.
- This is `Rostlab/prot_albert`, the UniRef100 model.
- Residues are tokenized space-separated, one SentencePiece `▁X` piece per residue between `[CLS]` and `[SEP]`; a sequence that does not tokenize that way is refused.
- The Jacobian runs in batches of 32 mutants, and the test's fixture (ubiquitin, 76 residues; lysozyme, 129) crosses that threshold many times over, 128 batches for the pair. Its map is checked against a reference computed with oplm's own functions at commit 704e01f on the same bf16 weights: exact symmetry, Pearson r >= 0.98, and top-L long-range (|i - j| >= 24) overlap >= 0.8, the tolerance for GPU kernel noise.
- With an NVIDIA driver older than 580 the GPU is not usable and the default device falls back to CPU, many times slower; `--device cuda` then fails rather than fall back.
- A large FASTA is one full Jacobian per sequence, quadratic in length: hours past a few hundred sequences, so submit it as a job, not an interactive `srun`.
