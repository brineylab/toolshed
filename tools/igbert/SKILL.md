---
name: igbert
description: "IgBert paired antibody BERT language model: pseudo-log-likelihoods, per-pair embeddings, masked infilling, and per-position amino-acid matrices for heavy/light antibody pairs. Reach for it to score, embed, or infill a paired antibody, or to compute its per-position mutation profile; use igbert-unpaired for a single chain."
metadata:
  version: 1.0.1
---
# igbert

Image: `ghcr.io/brineylab/toolshed/igbert:1.0.1`

IgBert (`transformers`' BERT implementation), a 30-layer, 1024-dimensional antibody language model with 420M fp32 parameters. Hugging Face `Exscientia/IgBert` revision `81a4abb8a4dac5cdcbefc28e5b331a82224c485d` is in the image as safetensors, and no network is needed. The caller supplies separate chain FASTAs; the tool constructs heavy–`[SEP]`–light input.

## Commands

```bash
igbert pll --heavy h.fasta --light l.fasta --out pll.tsv
igbert embed --heavy h.fasta --light l.fasta --out emb.npz
igbert fill --heavy masked-h.fasta --light masked-l.fasta --out filled.fasta
igbert matrix --heavy h.fasta --light l.fasta --out matrix.npz
```

Each takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** two FASTAs, heavy chains under `--heavy` and light chains under `--light`, matched by the first word of each id. Ids must be unique. Sequences contain the 20 standard amino acids; `fill` accepts `<mask>` at positions to infill. A pair has at most **297 residues**: the paper's paired preprocessing pads to 300 tokens including `[CLS]`, the chain `[SEP]` and final `[SEP]`. The checkpoint itself has 40,000 learned position embeddings, so this is the training-input limit, not a BERT truncation constraint.
- **`pll`:** TSV with header `id`, `pll`. Every residue in both chains is masked independently and the true residue's natural log-probability is summed. Divide by pair length when comparing lengths.
- **`embed`:** `.npz` with `ids` and float32 `(n, 1024)` `embeddings`: BERT's last hidden layer mean-pooled over all heavy and light residue positions, excluding all three separator/special positions.
- **`fill`:** FASTA with one record per pair. Heavy and light chains are joined by `|` only for readable output; this is not model input. Every `<mask>` is independently replaced by its highest-probability standard amino acid in one forward pass, while every unmasked residue remains unchanged.
- **`matrix`:** `.npz` with `ids`; `amino_acids`, the 20-element column order `ACDEFGHIKLMNPQRSTVWY`; and one float32 `(L, 20)` array per id. Every residue is masked independently in batches of 32, with all other positions unmodified. Values are the columns of the full-vocabulary masked-LM log-softmax at the 20 standard amino acids, without renormalizing over those columns; each row can therefore sum to less than probability one. Rows run from heavy residue 1 through light residue end, with no separator row.

## Compute

One NVIDIA GPU with driver 580 or newer (CUDA 13 image). The 1.68 GB safetensors checkpoint is loaded in fp32. `pll` and `matrix` cost one forward pass per residue, in batches of 32 masked pair copies; `embed` and `fill` cost one forward pass per pair. Each command prints measured model-load and compute seconds, residue throughput, GPU name and peak GPU memory; the hardware test record contains those measurements.

## Running it

On a laptop, from the directory holding the FASTAs:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/igbert:1.0.1 igbert matrix --heavy h.fasta --light l.fasta --out matrix.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/igbert:1.0.1 \
  --container-mounts="$PWD":/work --container-workdir=/work igbert pll --heavy h.fasta --light l.fasta --out pll.tsv
```

## Gotchas

- **The chain separator is directly evidenced, not guessed from vocabulary.** Kenlay et al., *Large scale paired antibody language models* (arXiv:2403.17889), §6 says paired inputs concatenate heavy and light with a separator, and Appendix A constructs `" ".join(sequence_heavy) + " [SEP] " + " ".join(sequence_light)` itself at https://github.com/Exscientia/IgBert/blob/main/README.md. Its tokenization example is `[CLS] heavy-residues [SEP] light-residues [SEP]`. The tool accepts no caller-written separator. §3/Table 1 documents the 300-token paired training target.
- IgBert is fine-tuned from a separate unpaired checkpoint. For single-chain scoring use the separate `igbert-unpaired` tool, not a fabricated empty light chain.
- The test uses the paper's first 228-residue example pair (119 heavy + 109 light), crossing the wrapper's 32-mask batching threshold seven times. Direct `BertForMaskedLM` per-position full-vocabulary log-softmax reference calls reproduce `matrix` and `pll`, the documented last-layer pooling reproduces `embed`, and independent argmax infilling reproduces `fill`. The load requires `missing_keys=[]`, `unexpected_keys=[]` and `mismatched_keys=[]`.
- The weights' upstream repository and metadata carry the **MIT license**, which permits redistribution with its copyright and permission notice. The license is shipped beside the weights as `/opt/model/LICENSE`; no credential is needed or stored. Exscientia's Hugging Face model API declares `license: mit`; the corresponding Zenodo deposit 10876909 independently offers the same weights under CC BY 4.0.
