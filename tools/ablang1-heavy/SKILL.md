---
name: ablang1-heavy
description: "AbLang1 heavy-chain antibody language model: pseudo-log-likelihoods, per-sequence embeddings, masked infilling, and per-position amino-acid matrices for antibody heavy chains. Reach for it to score, embed, or infill a heavy chain, or to compute its per-position mutation profile."
metadata:
  version: 1.0.0
---
# ablang1-heavy

Image: `ghcr.io/brineylab/toolshed/ablang1-heavy:1.0.0`

AbLang1 (`ablang` 0.3.1), a 12-layer, 768-dimensional antibody encoder with twine-bound fairseq attention, trained on OAS heavy chains. The official heavy weights are in the image at the pin verified below, and no network is needed. The separately trained light-chain model is `ablang1-light`.

## Commands

```bash
ablang1-heavy pll --in heavy.fasta --out pll.tsv
ablang1-heavy embed --in heavy.fasta --out emb.npz
ablang1-heavy fill --in masked-heavy.fasta --out filled.fasta
ablang1-heavy matrix --in heavy.fasta --out matrix.npz
```

Each takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** a FASTA of single heavy chains under `--in`. The first word of each header is its id; ids must be unique. Sequences contain the 20 standard amino acids; `fill` accepts `<mask>` at positions to infill. A chain has at most **128 residues**, the package's own heavy-chain preprocessing bound.
- **`pll`:** TSV with header `id`, `pll`. Every residue is masked independently and the true residue's natural log-probability is summed over positions. Divide by chain length when comparing lengths. AbLang1 has no package-level pseudo-log-likelihood function: this uses its masked language-model head directly at every position.
- **`embed`:** `.npz` with `ids` in input order and float32 `(n, 768)` `embeddings`: the last hidden layer mean-pooled over residue positions, excluding `<`, `>` and padding, as upstream `seqcoding` does.
- **`fill`:** FASTA with the same ids. Every `<mask>` is independently replaced by its highest-logit standard amino acid in one forward pass, exactly as upstream `restore` does; every unmasked residue remains unchanged.
- **`matrix`:** `.npz` with `ids`; `amino_acids`, the 20-element column order `ACDEFGHIKLMNPQRSTVWY`; and one float32 `(L, 20)` array per id. Every residue is masked independently in batches of 32, with all other positions unmodified. The model returns 20 amino-acid logits at each position (`AbLang(...)[..., 1:21]`); each row is a natural log-softmax over just those 20 columns in the usual matrix convention. No separator or start/end row is included.

## Compute

One NVIDIA GPU with driver 580 or newer (CUDA 13 image). The 343 MB PyTorch checkpoint is loaded in fp32. `pll` and `matrix` cost one forward pass per residue, in batches of 32 masked sequence copies; `embed` and `fill` cost one forward pass per sequence. Each command prints measured model-load and compute seconds, residue throughput, GPU name and peak GPU memory; the hardware test record `results/ablang1-heavy/1.0.0.txt` contains measurements on the 119-residue fixture.

## Running it

On a laptop, from the directory holding the FASTA:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/ablang1-heavy:1.0.0 ablang1-heavy matrix --in heavy.fasta --out matrix.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/ablang1-heavy:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work ablang1-heavy pll --in heavy.fasta --out pll.tsv
```

## Gotchas

- **Heavy and light are separate models, with no pair separator.** Upstream `ablang/pretrained.py` (`ablang==0.3.1`) exposes the `chain="heavy"` or `chain="light"` argument and its `ABtokenizer` constructs exactly `<` + one chain + `>`; no heavy/light pair constructor or separator exists. It sets the heavy training input limit to 128 and light to 127. Use this tool for heavy chains and `ablang1-light` for light chains; do not join them.
- **Pinned weights with a chain-specific name.** The weight archive `https://opig.stats.ox.ac.uk/data/downloads/ablang-heavy.tar.gz` is original upstream's direct download URL, because neither the package nor an HTTP revision endpoint supplies a revision hash. The build requires archive size 318,636,334 bytes and SHA-256 `7c939d690cadbd2464901f80f58bf7018a6cd2b7edd73ff2ba6eb29f97ffa4f7`; its `amodel.pt` SHA-256 is `ab0888f7dc5e89fbf10568a100f081eadf9386bd460d3cdf1d6731044a01266b`.
- The 119-residue fixture crosses the wrapper's 32-mask batching threshold three times (four masked per-position batches, the last containing 23 copies), near but below the 128-residue limit. Direct upstream `AbLang` masked-logit calls reproduce `matrix` and `pll`; an independent last-layer pooling call reproduces `embed`; and upstream `ablang.pretrained(..., mode="restore")` reproduces `fill` exactly. The package supplies raw rescoring logits, not normalized pseudo-likelihoods, so the test does not pretend otherwise.
- Upstream's code is **BSD 3-Clause**, which permits redistribution with the copyright notice and license retained. Traceability of the direct-download weight archive is limited to the model's official package and no separately published weight license was found; this is the same official-source status that lets redistribution proceed under the code's permissive license. The upstream license is shipped beside the weights as `/opt/ablang/LICENSE`; no credential is needed or stored.
