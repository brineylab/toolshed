---
name: ablang1-light
description: "AbLang1 light-chain antibody language model: pseudo-log-likelihoods, per-sequence embeddings, masked infilling, and per-position amino-acid matrices for antibody light chains. Reach for it to score, embed, or infill a light chain, or to compute its per-position mutation profile."
metadata:
  version: 1.0.0
---
# ablang1-light

Image: `ghcr.io/brineylab/toolshed/ablang1-light:1.0.0`

AbLang1 (`ablang` 0.3.1), a 12-layer, 768-dimensional antibody encoder with twine-bound fairseq attention, trained on OAS light chains. The official light weights are in the image at the pin verified below, and no network is needed. The separately trained heavy-chain model is `ablang1-heavy`.

## Commands

```bash
ablang1-light pll --in light.fasta --out pll.tsv
ablang1-light embed --in light.fasta --out emb.npz
ablang1-light fill --in masked-light.fasta --out filled.fasta
ablang1-light matrix --in light.fasta --out matrix.npz
```

Each takes `--device cuda` or `--device cpu`; the default is the GPU when one is visible.

## Inputs and outputs

- **In:** a FASTA of single light chains under `--in`. The first word of each header is its id; ids must be unique. Sequences contain the 20 standard amino acids; `fill` accepts `<mask>` at positions to infill. A chain has at most **127 residues**, the package's own light-chain preprocessing bound.
- **`pll`:** TSV with header `id`, `pll`. Every residue is masked independently and the true residue's natural log-probability is summed over positions. Divide by chain length when comparing lengths. AbLang1 has no package-level pseudo-log-likelihood function: this uses its masked language-model head directly at every position.
- **`embed`:** `.npz` with `ids` in input order and float32 `(n, 768)` `embeddings`: the last hidden layer mean-pooled over residue positions, excluding `<`, `>` and padding, as upstream `seqcoding` does.
- **`fill`:** FASTA with the same ids. Every `<mask>` is replaced by its highest-logit standard amino acid in one forward pass, exactly as upstream `restore` does; every unmasked residue remains unchanged. Multiple masked positions in the same sequence are therefore predicted jointly from that single pass, preserving the original non-masked context, as in the upstream `restore` path.
- **`matrix`:** `.npz` with `ids`; `amino_acids`, the 20-element column order `ACDEFGHIKLMNPQRSTVWY`; and one float32 `(L, 20)` array per id. Every residue is masked independently in batches of 32, with all other positions unmodified. The model returns 20 amino-acid logits at each position (`AbLang(...)[..., 1:21]`) in AbLang's own vocabulary order, `MRHKDESTNQCGPAVIFYWL`; the wrapper selects those columns by `ABtokenizer.vocab_to_token` to reorder them into the documented matrix convention. Each row is a natural log-softmax over those same 20 vocabulary columns. No separator or start/end row is included.

## Compute

One NVIDIA GPU with driver 580 or newer (CUDA 13 image). The 319 MB PyTorch checkpoint is loaded in fp32. `pll` and `matrix` cost one forward pass per residue, in batches of 32 masked sequence copies; `embed` and `fill` cost one forward pass per sequence. Each command prints measured model-load and compute seconds, residue throughput, GPU name and peak GPU memory; the hardware test record contains measurements on the 109-residue fixture.

## Running it

On a laptop, from the directory holding the FASTA:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/ablang1-light:1.0.0 ablang1-light matrix --in light.fasta --out matrix.npz
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/ablang1-light:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work ablang1-light pll --in light.fasta --out pll.tsv
```

## Gotchas

- **Heavy and light are separate models, with no pair separator.** Upstream `ablang/pretrained.py` (`ablang==0.3.1`) exposes the `chain="heavy"` or `chain="light"` argument and its `ABtokenizer` constructs exactly `<` + one chain + `>`; no heavy/light pair constructor or separator exists. It sets the heavy training input limit to 128 and light to 127. Use this tool for light chains and `ablang1-heavy` for heavy chains; do not join them.
- **Pinned weights with a chain-specific name.** The weight archive `https://opig.stats.ox.ac.uk/data/downloads/ablang-light.tar.gz` is original upstream's direct download URL, because neither the package nor an HTTP revision endpoint supplies a revision hash. The build requires archive size 318,291,943 bytes and SHA-256 `b5310bc75903feb781ef88adf3003f0f65595755755afa56b81c41a957c50cc9`; its `amodel.pt` SHA-256 is `b5b0deff394f2a799091627876d0113b7db12c026aee8d377fec1f9b53c56dfb`.
- The 109-residue IgBert example light-chain fixture crosses the wrapper's 32-mask batching threshold three times (four masked per-position batches, the last containing 13 copies), near but below the 127-residue limit. Direct upstream `AbLang` masked-logit calls, indexed by `ABtokenizer.vocab_to_token[residue]`, reproduce `matrix` and `pll`, and the fixture's mean per-residue PLL must exceed the uniform-residue bound `log(1/20)`; an independent last-layer pooling call reproduces `embed`; and upstream `ablang.pretrained(..., mode="restore")` reproduces `fill` exactly. The package supplies raw rescoring logits, not normalized pseudo-likelihoods, so the test does not pretend otherwise.
- Upstream's code is **BSD 3-Clause**, which permits redistribution with the copyright notice and license retained. Traceability of the direct-download weight archive is limited to the model's official package and no separately published weight license was found; this is the same official-source status that lets redistribution proceed under the code's permissive license. The upstream license is shipped beside the weights as `/opt/ablang/LICENSE`; no credential is needed or stored.
