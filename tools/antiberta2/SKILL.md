---
name: antiberta2
description: "AntiBERTa2 paired antibody RoFormer language model: pseudo-log-likelihoods, per-pair embeddings, masked infilling, and per-position amino-acid matrices for heavy/light antibody pairs. Reach for it to score, embed, or infill a paired antibody, or to compute its per-position mutation profile."
metadata:
  version: 1.0.3
---
# antiberta2

Image: `ghcr.io/brineylab/toolshed/antiberta2:1.0.3`

AntiBERTa2 (`transformers`' RoFormer implementation), a 16-layer, 1024-dimensional antibody language model. Hugging Face `alchemab/antiberta2` revision `3123350a3ee741c02a5dcf5efc771f3fb588ae77` is in the image, and no network is needed. The caller supplies chains separately; the tool constructs the model's empirically validated paired representation.

## Commands

```bash
antiberta2 pll --heavy h.fasta --light l.fasta --out pll.tsv
antiberta2 embed --heavy h.fasta --light l.fasta --out emb.npz
antiberta2 fill --heavy masked-h.fasta --light masked-l.fasta --out filled.fasta
antiberta2 matrix --heavy h.fasta --light l.fasta --out matrix.npz
```

Each takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** two FASTAs, heavy chains under `--heavy` and light chains under `--light`, matched by the first word of each id. Ids must be unique. Sequences contain the 20 standard amino acids; `fill` accepts `<mask>` at positions to infill. A pair has at most **251 residues** (256 model positions less `[CLS]`, two `[SEP]` tokens, `Ḣ` and `Ḷ`).
- **`pll`:** TSV with header `id`, `pll`. Every residue in both chains is masked independently and its true residue's natural log-probability is summed. Normalize by pair length when comparing lengths.
- **`embed`:** `.npz` with `ids` and float32 `(n, 1024)` `embeddings`: the RoFormer's last hidden layer mean-pooled over all heavy and light residue positions, excluding specials and chain markers.
- **`fill`:** FASTA with one record per pair. Heavy and light sequences are joined by `|` for readable output; this is not the model input separator. Every `<mask>` is replaced by its highest-probability standard amino acid in one forward pass, while every unmasked residue remains unchanged.
- **`matrix`:** `.npz` with `ids`; `amino_acids`, the 20-element column order `ACDEFGHIKLMNPQRSTVWY`; and one float32 `(L, 20)` array per id. Each pair's residues are masked one at a time in batches of 32, with the other positions unchanged. Rows are natural log-probabilities over the 20 standard amino acids and run from heavy residue 1 through light residue end.

## Compute

One NVIDIA GPU with driver 580 or newer (CUDA 13 image). The model is loaded in fp32 from its pinned safetensors checkpoint. `pll` and `matrix` cost one forward pass per residue, in batches of 32 masked pair copies; `embed` and `fill` cost one forward pass per pair. Each command prints measured model-load and compute seconds, residue throughput, B200 GPU name, and peak GPU memory; the hardware test record contains those measurements.

## Running it

On a laptop, from the directory holding the FASTAs:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/antiberta2:1.0.3 antiberta2 matrix --heavy h.fasta --light l.fasta --out matrix.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/antiberta2:1.0.3 \
  --container-mounts="$PWD":/work --container-workdir=/work antiberta2 pll --heavy h.fasta --light l.fasta --out pll.tsv
```

## Gotchas

- **Paired-input format is empirical, not inferred from token names.** The tool sends `Ḣ <heavy-residues> [SEP] Ḷ <light-residues>` to `RoFormerTokenizer`, which adds `[CLS]` and final `[SEP]`. In a three-pair probe at the pinned revision, masking `Ḣ` at the first position gave heavy-first log-odds 20.82, 16.92 and 18.85 for the correct marker; the second marker strongly preferred the already observed first-chain marker (-0.51, -0.17, -0.60 log-odds). Reversing chain order and markers reversed those predictions. Scored over all 228–230 residues, heavy-first mean PLL was -1.3390, -1.2516 and -1.2016 versus light-first -1.3514, -1.2595 and -1.2114. This establishes the tool's heavy-first marker order for the tested model, not a recovered detail of upstream's undocumented training pipeline.
- Marker token ids at the pinned vocabulary are `Ḣ`=26, `Ḷ`=27, `[CLS]`=1, `[SEP]`=2 and `[MASK]`=3. AntiBERTa2's published `config.json` has `max_position_embeddings=256`; the wrapper rejects longer pairs instead of allowing integer rotary positions to exceed it silently.
- The 228-residue fixture is upstream's example heavy/light pair, 121 heavy + 107 light residues. It crosses the wrapper's 32-mask batching threshold seven times; the independent check masks one residue per forward pass rather than reusing the wrapper's encoder.
- `matrix` compares every log-probability against independent Transformers RoFormer calls; `pll` must equal its true-residue sum, `embed` must match independent last-layer pooling, and `fill` must exactly match independent argmax infilling. The weight load requires `missing_keys=[]` and `mismatched_keys=[]`.
- Weights and source carry Alchemab's **AntiBERTa2 License**, which permits non-commercial research and academic use and redistribution with attribution, license inclusion and the required runtime notice. The upstream `LICENSE` is shipped beside the weights as `/opt/model/LICENSE`, and every invocation prints the notice to stderr. Commercial use is not permitted by this license; no credential is needed or stored.
