---
name: igbert
description: "IgBert paired antibody BERT language model: pseudo-log-likelihoods, per-pair embeddings, masked infilling, and per-position amino-acid matrices for heavy/light antibody pairs. Reach for it to score, embed, or infill a paired antibody, or to compute its per-position mutation profile."
metadata:
  version: 1.0.0
---
# igbert

Image: `ghcr.io/brineylab/toolshed/igbert:1.0.0`

IgBert, Exscientia's 30-layer, 1024-dimensional antibody BERT with 420M parameters, fine-tuned for paired chains from IgBert-unpaired. Hugging Face `Exscientia/IgBert` revision `81a4abb8a4dac5cdcbefc28e5b331a82224c485d` is in the image, and no network is needed. The caller supplies chains separately; the tool joins them with the model's own separator.

## Commands

```bash
igbert pll --heavy h.fasta --light l.fasta --out pll.tsv
igbert embed --heavy h.fasta --light l.fasta --out emb.npz
igbert fill --heavy masked-h.fasta --light masked-l.fasta --out filled.fasta
igbert matrix --heavy h.fasta --light l.fasta --out matrix.npz
```

Each takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** two FASTAs, heavy chains under `--heavy` and light chains under `--light`, matched by the first word of each id. Ids must be unique. Sequences contain the 20 standard amino acids; `fill` accepts `<mask>` at positions to infill. A pair has at most **297 residues**: the paired-training input length of 300 less `[CLS]` and three `[SEP]` tokens.
- **`pll`:** TSV with header `id`, `pll`. Every residue in both chains is masked independently and its true residue's natural log-probability is summed. Normalize by pair length when comparing lengths.
- **`embed`:** `.npz` with `ids` and float32 `(n, 1024)` `embeddings`: BERT's last hidden layer mean-pooled over heavy and light residue positions, excluding `[CLS]`, `[SEP]` and `[PAD]`.
- **`fill`:** FASTA with one record per pair. Heavy and light sequences are joined by `|` for readable output; this is not the model input separator. Every `<mask>` is replaced by its highest-probability standard amino acid in one forward pass, while every unmasked residue remains unchanged.
- **`matrix`:** `.npz` with `ids`; `amino_acids`, the 20-element column order `ACDEFGHIKLMNPQRSTVWY`; and one float32 `(L, 20)` array per id. Each pair's residues are masked one at a time in batches of 32, with the other positions unchanged. Rows are natural log-probabilities over the 20 standard amino acids and run from heavy residue 1 through light residue end.

## Compute

One NVIDIA GPU with driver 580 or newer (CUDA 13 image). The model is loaded in fp32 from its pinned safetensors checkpoint. `pll` and `matrix` cost one forward pass per residue, in batches of 32 masked pair copies; `embed` and `fill` cost one forward pass per pair. Each command prints measured model-load and compute seconds, residue throughput, GPU name, and peak GPU memory; the hardware test record contains those measurements.

## Running it

On a laptop, from the directory holding the FASTAs:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/igbert:1.0.0 igbert matrix --heavy h.fasta --light l.fasta --out matrix.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/igbert:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work igbert pll --heavy h.fasta --light l.fasta --out pll.tsv
```

## Gotchas

- **The separator is `[SEP]`, heavy first, with a final `[SEP]`.** Kenlay et al., *Large scale paired antibody language models* (arXiv:2403.17889v1), section 6 states that paired finetuning concatenates heavy and light chains with a separator, and Appendix A identifies IgBert's separator as `[SEP]`. The public reference tokenizer example (the HF repository's `README.md` at the pinned revision) constructs `' '.join(sequence_heavy) + ' [SEP] ' + ' '.join(sequence_light)` with `add_special_tokens=True`, yielding `[CLS] heavy [SEP] light [SEP]` and two BERT segment ids. The wrapper constructs that format internally; the caller never supplies a separator.
- This paired model also accepts the **single-chain and otherwise-heavy segments used in the alternative `igbert-unpaired` tool**, rather than adding an extra single-chain mode to these four commands.
- The published paired model's training target length is 300 residues, which the wrapper enforces after excluding the four special tokens. The pinned checkpoint has a 40,000-entry absolute-position table rather than a 300-position architectural ceiling. Inputs containing unsupported characters or pre-existing masks in `pll` and `matrix` are rejected rather than being silently tokenized to `[UNK]`.
- The 233-residue fixture is an antibody pair with 121 heavy + 112 light residues. It crosses the 32-mask batching threshold seven times; the independent reference masks one residue per forward pass rather than using the wrapper's batched encoder. Padded multi-pair batches are not exercised; pairs are evaluated one at a time.
- `matrix` must match independent Transformers calls at every position, and its true-residue columns must reproduce `pll`. `embed` must match independent last-layer pooling and `fill` must exactly match independent argmax infilling. Loading checks require empty missing-key and mismatched-key lists.
- IgBert's Hugging Face repository declares the **MIT** license; its official Zenodo weights redistribution (record 10876909) declares **CC-BY-4.0**. The upstream attribution, MIT text, and verbatim CC-BY-4.0 text are shipped as `/opt/model/LICENSE`, `/opt/model/NOTICE`, and `/opt/model/CC-BY-4.0.txt`, and an attribution is printed to stderr on every invocation. No credential is needed or stored.
