---
name: igbert-unpaired
description: "IgBert single-chain antibody BERT language model: pseudo-log-likelihoods, per-sequence embeddings, masked infilling, and per-position amino-acid matrices for individual antibody heavy or light chains. Reach for it to score, embed, or infill one chain, or to compute its per-position mutation profile. This tool takes one chain at a time; use igbert for a heavy/light pair."
metadata:
  version: 1.0.0
---
# igbert-unpaired

Image: `ghcr.io/brineylab/toolshed/igbert-unpaired:1.0.0`

IgBert unpaired (`transformers`' BERT implementation), a 30-layer, 1024-dimensional antibody language model with 420M fp32 parameters, fine-tuned from ProtBert-BFD on unpaired OAS antibodies. Hugging Face `Exscientia/IgBert_unpaired` revision `53ed1be4a09fd30ef3f21ba0c8aae778ccdfff74` is in the image as a 1.68 GB safetensors checkpoint, and no network is needed. Each sequence is one heavy or light chain; the tool constructs `[CLS] <one chain> [SEP]` itself.

## Commands

```bash
igbert-unpaired pll --in sequences.fasta --out pll.tsv
igbert-unpaired embed --in sequences.fasta --out emb.npz
igbert-unpaired fill --in masked.fasta --out filled.fasta
igbert-unpaired matrix --in sequences.fasta --out matrix.npz
```

Each takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** a FASTA of single chains under `--in`. The first word of each header is its id; ids must be unique. Sequences contain the 20 standard amino acids; `fill` accepts `<mask>` at positions to infill. A chain has at most **510 residues**, the unpaired paper's 512-token block length less `[CLS]` and final `[SEP]`. The checkpoint has 40,000 learned position embeddings, so this is the documented training-input limit rather than a checkpoint truncation constraint.
- **`pll`:** TSV with header `id`, `pll`. Every residue is masked independently and the true residue's natural log-probability is summed. Divide by chain length when comparing lengths.
- **`embed`:** `.npz` with `ids` and float32 `(n, 1024)` `embeddings`: BERT's last hidden layer mean-pooled over residue positions, excluding `[CLS]`, final `[SEP]` and padding, as the released unpaired model's average-residue representation example does.
- **`fill`:** FASTA with the same ids and one completed chain per input. Every `<mask>` is replaced by its highest-probability standard amino acid in one forward pass, while every unmasked residue remains unchanged.
- **`matrix`:** `.npz` with `ids`; `amino_acids`, the 20-element column order `ACDEFGHIKLMNPQRSTVWY`; and one float32 `(L, 20)` array per id. Every residue is masked independently in batches of 32, with all other positions unmodified. Values are the columns of the full-vocabulary masked-LM log-softmax at the 20 standard amino acids, without renormalizing over those columns; each row can therefore sum to less than probability one. There are no start/end rows.

## Compute

One NVIDIA GPU with driver 580 or newer (CUDA 13 image). The fp32 checkpoint is loaded from the image. `pll` and `matrix` cost one forward pass per residue, in batches of 32 masked sequence copies; `embed` and `fill` cost one forward pass per sequence. Each command prints measured model-load and compute seconds, residue throughput, GPU name and peak GPU memory; the hardware test record contains measurements on the 109-residue fixture.

## Running it

On a laptop, from the directory holding the FASTA:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/igbert-unpaired:1.0.0 igbert-unpaired matrix --in sequences.fasta --out matrix.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/igbert-unpaired:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work igbert-unpaired pll --in sequences.fasta --out pll.tsv
```

## Gotchas

- **Single-chain format is directly evidenced.** Kenlay et al., *Large scale paired antibody language models* (arXiv:2403.17889), §3 and §6 distinguish unpaired pre-training from paired fine-tuning; Appendix B's released sequence-average algorithm masks the tokenizer's special tokens. The released `Exscientia/IgBert_unpaired` example explicitly constructs single chains by `" ".join(sequence)` and shows `[CLS] E V V M [SEP]`; the tool reproduces this without any caller-written separator. Use the separate `igbert` tool for heavy–`[SEP]`–light pairs; do not concatenate two chains here.
- The 109-residue fixture is a released example light chain. It crosses the wrapper's 32-mask batching threshold three times (four masked batches, the last containing 13 copies). Direct `BertForMaskedLM` per-position full-vocabulary log-softmax calls reproduce `matrix` and `pll`, last-layer residue pooling reproduces `embed`, and independent argmax infilling reproduces `fill`; the mean per-residue PLL must exceed uniform-residue log-probability. The load requires `missing_keys=[]`, `unexpected_keys=[]` and `mismatched_keys=[]`.
- This is the distinct unpaired checkpoint, not a permissive way to send a single chain through paired `igbert`. Its Hugging Face card declares `license: mit`; the upstream Exscientia **MIT license** permits redistribution with its copyright and permission notice and is included beside the weights as `/opt/model/LICENSE`. No credential is needed or stored.
