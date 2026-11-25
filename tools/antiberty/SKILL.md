---
name: antiberty
description: "AntiBERTy single-chain antibody language model: pseudo-log-likelihoods, per-sequence embeddings, masked infilling, and per-position amino-acid matrices for antibody heavy or light variable domains. Reach for it to score, embed, or infill an antibody chain, or to compute its per-position mutation profile."
metadata:
  version: 1.0.5
---
# antiberty

Image: `ghcr.io/brineylab/toolshed/antiberty:1.0.5`

AntiBERTy (`antiberty` 1.0.0), an eight-layer BERT encoder with 512-dimensional representations, trained on 558M antibody variable domains. Each input sequence is one heavy or light chain; this tool does not concatenate heavy/light pairs. Weights are in the image at Hugging Face `jeffruffolo/AntiBERTy` revision `d582006e0c7dc6917d26fa60740a357e8df7b466`, and no network is needed.

## Commands

```bash
antiberty pll --in sequences.fasta --out pll.tsv
antiberty embed --in sequences.fasta --out emb.npz
antiberty fill --in masked.fasta --out filled.fasta
antiberty matrix --in sequences.fasta --out matrix.npz
```

Each takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible. `pll` and `matrix` default to `--normalize standard`; `--normalize full-vocab` is also accepted. For this model's vocabulary the two choices give identical values.

## Inputs and outputs

- **In:** a FASTA of single chains under `--in`, in the protein tools' convention. IDs must be unique. Sequences contain the 20 standard amino acids; `fill` additionally accepts `<mask>` at positions to infill. The model also accepts `_` as its mask spelling, but this interface consistently reads and writes FASTA. There is no pair separator: do not pass joined heavy/light sequences.
- **`pll`:** TSV with header `id`, `pll`. Each amino acid is masked in turn; the natural log-probability of its true residue is summed over the chain. Negative values closer to zero are more likely. Divide by residue count to compare different lengths. Under either normalization choice, the sum equals upstream's package `pseudo_log_likelihood` mean multiplied by chain length.
- **`embed`:** `.npz` with `ids` in input order and `embeddings`, float32 `(n, 512)`: the last hidden layer mean-pooled over residue positions, excluding `[CLS]`, `[SEP]` and `[PAD]`.
- **`fill`:** FASTA with one completed sequence per input ID. Every mask is independently replaced by the highest-probability standard amino acid; every unmasked residue remains unchanged.
- **`matrix`:** `.npz` with `ids`; `amino_acids`, the 20-element column order `ACDEFGHIKLMNPQRSTVWY`; and one float32 `(L, 20)` array per ID. Each residue is masked independently with all other positions unmodified. Entries are natural log-probabilities of all 20 standard amino acids at that position, normalized over those 20 columns. Both normalization choices give the same rows because upstream masks its remaining five, special-token logits to minus infinity.

## Compute

One GPU, one sequence at a time for `pll` and `matrix`, in batches of 32 masked copies. Each of those commands costs one masked forward pass per residue; `embed` and `fill` cost one forward pass per input (a single padded batch for `fill`). The model holds about 26M fp32 parameters, with a 104 MB safetensors checkpoint. Measured per-command timings, throughput and GPU memory are printed by every command and recorded in `results/antiberty/1.0.5.txt` for the 121-residue fixture. The positional limit is **510 residues** plus `[CLS]` and `[SEP]`; longer inputs are rejected.

## Running it

On a laptop, from the directory holding the FASTAs (omit `--device` without a GPU):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/antiberty:1.0.5 antiberty matrix --in sequences.fasta --out matrix.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/antiberty:1.0.5 \
  --container-mounts="$PWD":/work --container-workdir=/work antiberty pll --in sequences.fasta --out pll.tsv
```

## Gotchas

- **Single-chain only.** Upstream's `antiberty/tokenizer.py` at source release `1f7478450dee58557185b8d5ca0f1eb32bd3c30c` (`antiberty` 1.0.0) adds exactly `[CLS] <one sequence> [SEP]`, and `AntiBERTyRunner` passes one sequence at a time through that tokenizer. Unlike AbLang2, this package has no heavy/light pair constructor to adopt. A caller must score heavy and light individually rather than infer a paired format.
- **Normalization matches upstream.** Upstream's `pseudo_log_likelihood` sets its five special-token logits to `-inf` before applying `log_softmax`. Its vocabulary is those five special tokens plus the 20 reported amino acids, so no probability remains outside the matrix's 20 columns. `--normalize standard` and `--normalize full-vocab` therefore give the same values, equal to upstream. The test runs `pll --normalize full-vocab` and compares its sum with upstream `AntiBERTyRunner.pseudo_log_likelihood` multiplied by chain length; it also verifies that the default matrix's true-residue column sum reproduces that same PLL and that every row sums to probability one.
- **Weights are not loaded permissively.** The test separately calls `AntiBERTy.from_pretrained(..., output_loading_info=True)` and requires `missing_keys == []` instead of depending on an auto-masked-language-model's silently initialized heads. The checkpoint's species, chain and graft heads are reported as unexpected only when a plain BERT loader is used, not by this full upstream model class. The test also matches upstream last-layer mean embeddings and exact masked infilling.
- The 121-residue fixture crosses the wrapper's 32-mask batching threshold three times (four masked batches, the last containing 25 copies). It tests ordinary antibody variable-domain lengths, not the model's 510-residue positional limit or accuracy on full-length antibody chains and non-antibody proteins.
- Input masks are deliberately rejected by `pll` and `matrix`; they are meaningful only to `fill`. Unsupported residues are rejected rather than silently mapped to `[UNK]`.
- Weights and code are MIT licensed; Hugging Face declares `license: mit` and upstream `LICENSE.MD` permits redistribution and requires retention of the copyright and license notice. Upstream's license is shipped beside the weights as `/opt/AntiBERTy/LICENSE`; no credential is needed or stored.
