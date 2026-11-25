---
name: msa-pairformer
description: "MSA Pairformer on Blackwell: A3M filtering, hhfilter diversity subsampling, query-only conditional (L, 20) logits in sharded Parquet, and C-beta contact score maps. Reach for it to extract masked-query teacher logits for protein language-model distillation."
metadata:
  version: 1.0.2
---
# msa-pairformer

Image: `ghcr.io/brineylab/toolshed/msa-pairformer:1.0.2`

`yoakiyama/MSA-Pairformer` at revision `7563e77a87536b5572f91683c39073ef348639a4`, with code commit `875363570df1ae484cf725aba382790444223005`, PyTorch 2.12.1 on CUDA 13 and cuEquivariance 0.10.0 for Blackwell. The cuEx main checkpoint and two contact-head checkpoints are baked into `/opt/model`; no network is used at run time.

## Commands

For a directory of single-chain `.a3m` or `.a3m.gz` files, each with its ungapped uppercase query in row 0:

```bash
msa-pairformer --in msas/ --out teacher_logits/ --depth 128 --k 7 --min-coverage 0.8
```

Prevent copy-paste leakage by dropping homologs above 95% identity to the query:

```bash
msa-pairformer --in msas/ --out capped_logits/ --depth 128 --k 7 --min-coverage 0.8 --query-identity-cap 0.95
```

For two independent, disjoint masks covering approximately 15% of query positions each:

```bash
msa-pairformer --in msas/ --out fast_logits/ --depth 128 --k 2 --min-coverage 0.8
```

Split a corpus among jobs and resume a completed subset:

```bash
msa-pairformer --in msas/ --out teacher_logits/ --depth 256 --k 7 --job-index 0 --num-jobs 8
```

Both commands' examples run the same file-driven command. `--device cpu` is available for debugging rather than production corpus extraction.

## Inputs and outputs

- **In:** A3M and gzip-compressed A3M files. Lowercase insertions and dots are removed before parsing. Row 0 must be the ungapped uppercase query; all other rows are cropped to its match columns, so their first and last positions can be gaps. Rows below `--min-coverage` are discarded after that crop. `--query-identity-cap` then removes homologs whose matching non-gap fraction at query columns exceeds the cap, leaving row 0 intact. The cap is deliberately independent of hhfilter's pairwise diversity filter.
- **Preprocessing:** upstream `MSA` preprocessing applies HHsuite 3.3.0 `hhfilter -M a3m -id 90 -cov 70 -qid 30 -qsc -20 -diff <depth>`, followed by upstream's greedy diversity selection when needed. Set `--depth 64`, `128`, or `256` for the desired maximum row count; shallow alignments keep fewer rows. Mask permutations and upstream preprocessing start from `--seed 42` by default. Metrics include `depth`, the row count after coverage filtering but before any clone cap, `neff`, `neff_sqrt_l`, original selected row indices, maximum pre-hhfilter query identity and inference/preprocessing time. Neff uses inverse-count weights above 80% pairwise identity, with gaps excluded and gap-gap matches not counted, on the selected MSA.
- **Per input `<id>`:** `teacher_logits/<id>/logits.parquet` holds one row per query position: `id`, zero-based `position`, `logits_f16`, `target_index`, `fold`, `is_valid_target`, and `has_conditional_logits`. `logits_f16` is a 40-byte little-endian float16 array, decoded by `numpy.frombuffer(value, dtype='<f2')`: 20 raw, pre-softmax channels in `ARNDCEQGHILKMFPSTWYV`, the upstream model's residue-token order. This is not alphabetical amino-acid order. `target_index=255` marks a noncanonical query residue. `mask.json` records the weights revision, all fold sizes and filtering settings. Only folds run with `k=2` have logits: `has_conditional_logits` is false on the remaining ~70% of positions. No unmasked-query predictions enter this file.
- **Contacts:** `<id>/contacts.npz` holds the first query-masked fold's float16 `(L,L)` predicted-C-beta-contact score map, symmetrized by the model's logistic-regression head at layer 15. C-beta scores remain defined at glycine, as in the model's original head. Distillation targets and contacts are artifacts of separate conditioning information; contacts do not contribute to the conditional perplexity metrics.
- **Metrics:** `metrics.jsonl`, one JSON per completed chain, includes raw NLL and entropy sums (`nats`), correct count and evaluated count, peak GPU allocations and per-fold seconds. Divide by `ncount` for perplexity `exp(nll_sum / ncount)`, entropy and accuracy. Per-chain JSON also serves as the resumability ledger; only complete chains enter `metrics.jsonl`. Keep separate configurations in separate `--out` directories so there is no mixing of settings.
- **Sharding:** `--num-jobs N --job-index I` selects every N-th sorted input. Use one output parent per job (`out/job_0`, etc.); then all `out/job_*/` directories form the corpus's logical shards. Within each directory the one-file-per-input layout is chosen to avoid multi-gigabyte Parquet row groups and allow per-chain resume. Complete destinations are rewound cleanly on failure.

## Compute

One Blackwell B200 is the supported inference target. CUDA 13 carries Blackwell sm_100 kernels; CPU use is a debugging path, not a throughput option. K=7 uses seven full encoder passes, K=2 two. The selected depth, query length and triangle multiplicative updates determine peak memory. Model load and real hw-filter timing affect short MSAs, so compare chains using the per-fold seconds, not total process startup. Actual B200 rates and peak memory are written per chain in `metrics.jsonl`. The command loads the model once for many MSAs. Production runtimes should be measured on this version before a corpus-size GPU-hour estimate.

## Running it

On a laptop with a Blackwell GPU:

```bash
podman run --rm --network none --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/msa-pairformer:1.0.2 \
  msa-pairformer --in msas/ --out teacher_logits/ --depth 128 --k 7 --min-coverage 0.8
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/msa-pairformer:1.0.2 \
  --container-mounts="$FLYLOFT_CAMPAIGN/data:/data" --container-workdir=/data \
  msa-pairformer --in /data/msas --out /data/teacher_logits \
  --depth 128 --k 7 --min-coverage 0.8 --query-identity-cap 0.95
```

## Gotchas

- A query with lowercase characters or match-column gaps is refused, rather than silently changing its PDB-to-residue alignment. Trim an end gapped A3M query to its ungapped match interval first; the caller must retain that interval when mapping contact maps to a PDB.
- The model's `lm_head` emits 26 channels. The Parquet bytes contain exactly the first 20 canonical channels; gaps, pad, mask and auxiliary amino-acid codes are dropped. Softmax of those 20 logits is the distillation distribution over this output alphabet.
- Masking only changes row 0, never any homolog, and every position's logits come from a pass where that query position was masked. Gaps and noncanonical true residues are excluded from assessment but remain sequence inputs.
- `--k 7` partitions every query position into seven balanced folds. `--k 2` uses two disjoint approximately 15% stretches of a seeded permutation, not two partitions of half a sequence. Other positive K values mean balanced full query partitions.
- The standard hhfilter settings retain upstream defaults, not the hidden `qid=0/qsc` variants used in some training scripts. The query clone cap is applied before hhfilter; its metadata max identity therefore reports filtering before hhfilter's own max-90% representative selection.
- Neff is computed from filtered selected rows, not the unfiltered database MSA. Use the same depth and identity-cap for extraction and any Neff gate, unless deliberately studying a full-MSA Neff.
- A test fixture of 120 residues and 40 A3M homolog rows crosses the real hhfilter and K-fold paths. Its selected source row 33 carries lowercase insertions and a dot, so the test exercises per-row insertion removal before query-column projection as well as dtype, coverage, no-homolog masking, row filtering, source-token target indexing, and C-beta map values and shape. It does not establish contact accuracy on experimental structures or quantify the longest-chain B200 memory use; these are benchmark measurements, not promises of the synthetic smoke test.
- The checkpoint's modified MIT license (weights, Pizza edition) permits redistribution and requires copyright/permission inclusion; code has the corresponding Coffee edition. Both license texts are at `/opt/model/LICENSE` and `/opt/model/CODE_LICENSE`. Cite MSA Pairformer (`Akiyama et al.`, Cell 2026, DOI `10.1016/j.cell.2026.06.029`) in derivative analyses.
