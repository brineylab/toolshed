---
name: ablang2
description: "AbLang2 paired antibody language model: pseudo-log-likelihoods, per-sequence embeddings, masked infilling, and per-position amino-acid matrices for heavy/light antibody pairs. Reach for it to score, embed, or infill a paired antibody, or to compute its per-position mutation profile."
metadata:
  version: 1.0.8
---
# ablang2

Image: `ghcr.io/brineylab/toolshed/ablang2:1.0.8`

AbLang2 paired (`ablang2` 0.2.1; Zenodo record `10185169`, the `ablang2-weights.tar.gz` published there) with 480-dimensional representations. The caller supplies heavy and light sequence FASTAs; the tool constructs the model's own paired representation. Weights are in the image, and no network is needed.

## Commands

```bash
ablang2 pll --heavy h.fasta --light l.fasta --out pll.tsv
ablang2 embed --heavy h.fasta --light l.fasta --out emb.npz
ablang2 fill --heavy masked-h.fasta --light masked-l.fasta --out filled.fasta
ablang2 matrix --heavy h.fasta --light l.fasta --out matrix.npz
```

Each takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** two FASTAs, heavy chains under `--heavy` and light chains under `--light`, matched by sequence id (the first word of each header). Ids must be unique. Sequences contain the 20 standard amino acids; `fill` accepts `<mask>` (or upstream's `*`) at positions to infill. Paired input is always heavy first, light second, encoded by the tool as `<heavy>|<light>`. The separator is `|`, vocabulary token id 25, a special token excluded from scoring and pooling.
- **`pll`:** TSV with header `id`, `pll`. Each amino acid in both chains is masked in turn; the natural log-probability of its true residue is summed over all residues. Negative values closer to zero are more likely. As in esm2's tool, divide by total residue count to compare different lengths; upstream's mean equals this sum divided by residue count.
- **`embed`:** `.npz` with `ids` in heavy-file order and `embeddings`, float32 `(n, 480)`: AbLang2's last-layer hidden states mean-pooled over all 20-standard-residue positions in both chains (separators, chain boundary tokens, and masks excluded).
- **`fill`:** FASTA with one record per pair. The heavy and light sequences are joined by `|`; every mask is replaced independently by the highest-probability standard amino acid (AbLang2's `<mask>` is represented as `*` in its vocabulary). Non-mask residues and pair order are preserved.
- **`matrix`:** `.npz` with `ids`; `amino_acids`, the 20-element column order `ACDEFGHIKLMNPQRSTVWY`; and one float32 `(L, 20)` array per id. Each amino acid in both chains is masked independently, with all other heavy and light positions unmodified; the entries are natural log-probabilities of all 20 standard amino acids at that position. Rows run from heavy residue 1 through light residue end (no separator row). Logits are normalized over the reported 20 standard amino acids.

The `fill` input fixtures exercise both lowercase `<mask>` and upstream's `*`; the output includes the model's `|` between chains, not the chain boundary tokens.

## Compute

One GPU, any of the cluster's B200 or other NVIDIA GPUs with driver 580 or newer (CUDA 13 image). The tool loads a 168 MB weight archive and keeps fp32 weights. `matrix` and `pll` each cost one forward pass per residue over both chains; sequences are scored one pair at a time in batches of 32 positions. Test-time timings and memory are recorded in `results/ablang2/1.0.8.txt` with the full fixture outputs. `embed` and `fill` require one forward pass per pair.

## Running it

On a laptop, from the directory holding the FASTAs (drop `--device` without a GPU):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/ablang2:1.0.8 ablang2 matrix --heavy h.fasta --light l.fasta --out matrix.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/ablang2:1.0.8 \
  --container-mounts="$PWD":/work --container-workdir=/work ablang2 pll --heavy h.fasta --light l.fasta --out pll.tsv
```

## Gotchas

- The 233-residue test fixture is upstream's example heavy/light pair (121 heavy and 112 light residues). It crosses the wrapper's 32-mask batching threshold seven times. Its `pll` is independently recomputed by calling upstream `pretrained([[heavy, light]], mode='pseudo_log_likelihood')` in the same image and checked against the tool within 0.02 nats total; upstream uses mean while the tool reports the sum. Matrix rows are normalized log-probabilities, and the true-residue column sums must reproduce `pll`.
- Upstream separator evidence is package source, not a model card: `ablang2/pretrained.py:add_extra_tokens` emits `f"<{heavy}>|<{light}>"`; `models/ablang2/tokenizers.py:ABtokenizer.encode` uses the same ordering and sets `sep_token = aa_to_token['|']`; `models/ablang2/vocab.py` maps `|` to 25. This heavy-first source pinning is why callers give chains separately rather than a joined string.
- There is no learned maximum sequence length in this rotary-embedding model. The fixture measures short/paired behavior rather than claiming an unbounded-input guarantee; memory and attention cost grow with total paired length.
- The test's comparison uses the same `0.2.1` package source and pinned weights but exercises upstream's own reduction and input formatting independently. CPU and GPU arithmetic can differ by small floating-point tolerances.
- Weights redistribution is permitted by Zenodo record `10185169`'s **BSD-3-Clause-Clear** license and the upstream AbLang2 repository's BSD 3-Clause license. The upstream code `LICENSE` and the weights' BSD-3-Clause-Clear text are shipped beside the weights at `/opt/ABLANG-paired/LICENSE` and `/opt/ABLANG-paired/LICENSE.weights`; no credential is needed or stored.
