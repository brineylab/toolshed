---
name: protbert
description: "ProtBERT (Rostlab/prot_bert, the UniRef100 model) contact maps from the categorical Jacobian, computed exactly as oplm computes them. Reach for it to turn protein sequences into long-range contact score maps comparable with OPLM's precision@L curves."
metadata:
  version: 1.0.4
---
# protbert

Image: `ghcr.io/brineylab/toolshed/protbert:1.0.4`

ProtBERT, the 420M-parameter BERT masked language model trained on UniRef100 (`Rostlab/prot_bert`: 30 layers, hidden size 1024, vocab of 30 tokens), run through `transformers`. The weights are in the image; it needs no network.

## Commands

```bash
protbert contacts --in seqs.fasta --out contacts.npz
```

Takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** FASTA, one or more protein sequences in the 20 canonical amino acids, up to 818 residues each. The first word of each header is the sequence's id. A sequence that is longer, or holds a non-canonical residue, is refused, naming it.
- **Out:** an `.npz` with one `(L, L)` float32 array per id, `L` being that sequence's length: the residue-only contact score map, with no rows or columns for `[CLS]`/`[SEP]`. Higher score means a more likely contact. The map is oplm's categorical-Jacobian contact score, the same computation as `oplm`'s structure eval (`compute_categorical_jacobian` + `categorical_jacobian_to_contact_map` at commit 704e01f, copied verbatim into the wrapper): the weights load and the forward passes run in bf16 (never INT8 or any other quantization), logits are cast to fp32, every position is substituted with every one of the 20 amino acids in oplm's `CANONICAL_AMINO_ACIDS` order to build the `(L, 20, L, 20)` Jacobian of logit deltas, which is then centered over all four axes, symmetrized, reduced by the square root of the sum of squares over the two amino-acid axes, diagonal-zeroed, APC-corrected, and symmetrized.

## Compute

One GPU, any of the cluster's, with NVIDIA driver 580 or newer (the image is CUDA 13); about 5 GB of GPU memory at the longest sequence. The Jacobian costs one forward pass per position per amino acid, in batches of 32 mutants, so its cost grows with the square of the length: on an A100, 2.7 s for a 76-residue sequence and 2.0 s at 129 residues (the first of a run pays CUDA warmup), and about 2 min at 818, extrapolated from that square law; loading the model adds about 15 s, and the first run on a machine also assembles and sha256-verifies the checkpoint from its committed chunks, a few seconds of I/O cached beside the system's temp directory thereafter. A few short sequences run on CPU too, many times slower.

## Running it

On a laptop, from the directory holding `seqs.fasta` (drop `--device` without a GPU):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/protbert:1.0.4 protbert contacts --in seqs.fasta --out contacts.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/protbert:1.0.4 \
  --container-mounts="$PWD":/work --container-workdir=/work protbert contacts --in seqs.fasta --out contacts.npz
```

## Gotchas

- The weights are the ProtTrans pretrained models, which the ProtTrans project releases under the Academic Free License v3.0; the license is shipped beside the weights at `/opt/model/LICENSE`, committed in `model/`. Cite Elnaggar et al., ProtTrans (IEEE TPAMI 2021) when you use them.
- The image ships the fp32 `pytorch_model.bin` (1.7 GB) of `Rostlab/prot_bert` @ `7a894481` as thirteen `128 MiB` chunks, `/opt/model/pytorch_model.bin.part00`–`.part12`, each fetched by one short build step and sha256-pinned (`fetch_chunk.py` carries the manifest); on its first run the wrapper assembles them, checks the whole file's sha256 (`6ea3edd2…`), and loads it with `dtype=bfloat16` — bit for bit the same bf16 weights a build-time conversion would have shipped, verified on CPU in a campaign run comparing the cast-at-load state dict and a forward pass against the converted file, so no quantization is involved and the reference still matches. Chunked because eight single-step builds of this image died on the machine that builds it (connections to huggingface.co that stall, then silent deaths in or just after the long weights step): short steps keep each committed layer small, and since the RUN instructions never change, a failed build's completed chunks stay in the builder's layer cache for the next attempt. The five config, tokenizer and license files are committed in `model/` too.
- This is `Rostlab/prot_bert`, the UniRef100 model, not `prot_bert_bfd`.
- Residues are tokenized space-separated, one token per residue between `[CLS]` and `[SEP]`; a sequence that does not tokenize that way is refused.
- The Jacobian runs in batches of 32 mutants, and the test's fixture (ubiquitin, 76 residues; lysozyme, 129) crosses that threshold many times over, 129 mutation batches for the pair (48 and 81). Each map is checked against `test/ref.npz`, a reference computed with oplm's own functions at commit 704e01f on the same bf16 weights: exact symmetry, Pearson r >= 0.98, and top-L long-range (|i - j| >= 24) overlap >= 0.8, the tolerance for GPU kernel noise.
- With an NVIDIA driver older than 580 the GPU is not usable and the default device falls back to CPU, many times slower; `--device cuda` then fails rather than fall back.
- A large FASTA is one full Jacobian per sequence, quadratic in length: hours past a few hundred sequences, so submit it as a job, not an interactive `srun`.
