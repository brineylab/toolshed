---
name: igbert-unpaired
description: "IgBert-unpaired single-chain antibody BERT language model: pseudo-log-likelihoods, per-sequence embeddings, masked infilling, and per-position amino-acid matrices for antibody heavy or light variable domains. Reach for it to score, embed, or infill a single antibody chain, or to compute its per-position mutation profile."
metadata:
  version: 1.0.0
---
# igbert-unpaired

Image: `ghcr.io/brineylab/toolshed/igbert-unpaired:1.0.0`

IgBert-unpaired (`transformers`' BERT implementation), a 30-layer, 1024-dimensional antibody language model with 420M fp32 parameters. Hugging Face `Exscientia/IgBert_unpaired` revision `53ed1be4a09fd30ef3f21ba0c8aae778ccdfff74` is in the image as safetensors, and no network is needed. Its separately trained paired counterpart is `igbert`.

## Commands

```bash
igbert-unpaired pll --in sequences.fasta --out pll.tsv
igbert-unpaired embed --in sequences.fasta --out emb.npz
igbert-unpaired fill --in masked.fasta --out filled.fasta
igbert-unpaired matrix --in sequences.fasta --out matrix.npz
```

Each takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** a FASTA of single chains under `--in`, one heavy or light chain per record. The first word of each header is the sequence's id; ids must be unique. Sequences contain the 20 standard amino acids; `fill` accepts `<mask>` at positions to infill. A chain has at most **198 residues**: the paper's unpaired preprocessing pads to 200 tokens including `[CLS]` and final `[SEP]`. The checkpoint itself has 40,000 learned position embeddings, so this is the training-input limit, not a BERT truncation constraint.
- **`pll`:** TSV with header `id`, `pll`. Every residue is masked independently and the true residue's natural log-probability is summed. Divide by residue count when comparing lengths.
- **`embed`:** `.npz` with `ids` in input order and float32 `(n, 1024)` `embeddings`: BERT's last hidden layer mean-pooled over residues, excluding `[CLS]` and final `[SEP]`.
- **`fill`:** FASTA with the same ids. Every `<mask>` is independently replaced by its highest-probability standard amino acid in one forward pass, while every unmasked residue remains unchanged.
- **`matrix`:** `.npz` with `ids`; `amino_acids`, the 20-element column order `ACDEFGHIKLMNPQRSTVWY`; and one float32 `(L, 20)` array per id. Every residue is masked independently in batches of 32, with all other positions unmodified. Values are the columns of the full-vocabulary masked-LM log-softmax at the 20 standard amino acids, without renormalizing over those columns; each row can therefore sum to less than probability one.

## Compute

One NVIDIA GPU with driver 580 or newer (CUDA 13 image). The 1.68 GB safetensors checkpoint is loaded in fp32. `pll` and `matrix` cost one forward pass per residue, in batches of 32 masked sequence copies; `embed` and `fill` cost one forward pass per sequence. Each command prints measured model-load and compute seconds, residue throughput, GPU name and peak GPU memory; the hardware test record contains those measurements.

## Running it

On a laptop, from the directory holding the FASTAs:

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

- **Single-chain only.** Kenlay et al., *Large scale paired antibody language models* (arXiv:2403.17889), §5 describes the unpaired pretraining and Table 1 its 200-token target. Its model usage source at https://github.com/Exscientia/IgBert_unpaired/blob/main/README.md constructs one space-separated chain at a time and documents `[CLS] chain-residues [SEP]`. There is no chain separator in this tool; score heavy and light individually. The separate `igbert` tool instead uses heavy–`[SEP]`–light.
- The 119-residue test fixture is the paper's first heavy-chain example. It crosses the wrapper's 32-mask batching threshold three times. Direct `BertForMaskedLM` per-position full-vocabulary log-softmax reference calls reproduce `matrix` and `pll`, the documented last-layer pooling reproduces `embed`, and independent argmax infilling reproduces `fill`. The load requires `missing_keys=[]`, `unexpected_keys=[]` and `mismatched_keys=[]`.
- The weights' upstream repository and metadata carry the **MIT license**, which permits redistribution with its copyright and permission notice. The license is shipped beside the weights as `/opt/model/LICENSE`; no credential is needed or stored. Exscientia's Hugging Face model API declares `license: mit`; the corresponding Zenodo deposit 10876909 independently offers the same weights under CC BY 4.0.
