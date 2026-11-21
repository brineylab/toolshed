---
name: stok
description: "STok's GCP-VQVAE Large structure tokenizer: turns protein structures into discrete structure tokens (one of 4096 per residue) from mmCIF text in Parquet shards or from structure files, as sharded Parquet with tokenizer-identity metadata. Reach for it to build a structure-token dataset over structures you hold, or to tokenize structures alongside their sequences for paired sequence-structure training."
metadata:
  version: 1.0.0
---
# stok

Image: `ghcr.io/brineylab/toolshed/stok:1.0.0`

STōk (github.com/briney/stok, MIT) at commit `5dcecfedd1040f30ca28de54ac6a6b22b9dc184b`, on PyTorch 2.14.1 (the base's CUDA 13.0 build), with the GCP-VQVAE **Large** preset's weights (huggingface.co/Mahdip72/gcp-vqvae-large, revision `64bb4ff628f7d2ccba587cdc9f0eaa97c1f3f9a1`, upstream MIT) baked into the image at `/opt/stok-gcp`, fetched at build time through STōk's own resolver, which verifies the checkpoint's pinned sha256 — so a run reaches no network and carries exactly the release every output shard's metadata names. The same code also decodes tokens back to backbone coordinates (`load_pretrained_decoder`), which is how the tool's test checks the tokens mean the structure.

## Commands

Parquet shards of mmCIF text — afdb-rep's shape: an `id` column, the mmCIF text in a `cif` column, and any columns to carry along:

```bash
stok tokenize-structures input_dir output_dir --preset large --text-column cif \
  --id-column id --keep-column mean_plddt --keep-column uniprot_accession \
  --include-coordinates --device cuda
```

A directory of structure files (`.cif`/`.mmcif` or `.pdb` with SEQRES; PDBs without SEQRES are refused):

```bash
stok tokenize-structures structures_dir output_dir --preset large --device cuda
```

Shard-parallel jobs — job `I` of `N` takes every N-th input shard, and finished output shards are skipped, so a rerun resumes and a requeued twin does no harm:

```bash
stok tokenize-structures input_dir output_dir --preset large --text-column cif \
  --num-jobs 16 --job-index 3 --device cuda
```

`--chain` selects one chain from each structure (default: the single polymer chain); `--rows-per-shard` (file mode only) sets output shard size; `--device cpu` works, slowly. `stok tokenize-structures --help` lists every option.

## Inputs and outputs

- **In (Parquet mode):** shards of rows shaped like afdb-rep's: a non-null `id` (or `--id-column`) and mmCIF text in `cif` (or `--text-column`); `--keep-column` names columns to carry into the output. The mmCIF needs a contiguous `entity_poly_seq` for its polymer and `atom_site` rows — exactly what an AlphaFold DB or experimental mmCIF carries.
- **Out:** one Parquet shard per input shard, named for its stem (`00000.parquet` → `00000.parquet`), zstd, holding `sequence_id`, `sequence` (STōk's own read of the deposited sequence), `structure_tokens` (a list of int16, one per residue: `null` where the residue lacks a complete backbone), `coordinates` when `--include-coordinates` (N/CA/C per residue, `null` at unobserved atoms), then the kept columns. Every shard's footer carries a `stok.tokenizer` metadata key: `preset`, `checkpoint_sha256`, `codebook_sha256`, `codebook_size`, `stok_version` — the identity downstream training should pin. Rows the fixed policy refuses land in `rejected/<shard>.jsonl` as `{"sequence_id", "reason"}` (a chain under 25 or over 1280 residues, `chains_too_short`/`chains_too_long`; unparseable or ambiguous structures, `parse_failed`/`metadata_conflict`/...); a shard whose every row is rejected is still written, empty, so shard counts stay stable.

## Compute

One GPU, or a CPU at perhaps a hundredth the speed. The 2.5 GB checkpoint loads once per process in about a minute, and inference needs a few GB of GPU memory at any chain length. Inference is FP32, one structure at a time — deliberate, so token identity never depends on batching or sharding (see Gotchas) — which caps throughput near a second per structure on a single GPU: the reference experiment measured 0.76 s per structure with the smaller Lite preset on an AMD 8060S, and this tool's test tokenizes its two 243-residue fixtures inside a minute on one B200 after the load. A corpus of millions of structures is a many-GPU-hour run: split it with `--num-jobs`/`--job-index` and budget from the per-structure rate; the campaign that built this tool measured Large on B200s at scale in its `afdb-rep-stok-large` dataset skill.

## Running it

On a laptop, from the directory holding the input shards:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/stok:1.0.0 \
  stok tokenize-structures input_dir output_dir --preset large --text-column cif \
  --id-column id --include-coordinates --device cuda
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/stok:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work \
  stok tokenize-structures input_dir output_dir --preset large --text-column cif \
  --id-column id --include-coordinates --device cuda
```

## Gotchas

- The tokenization policy is fixed — full chains of 25–1280 residues, native sequence inputs, reference imputation, FP32 — and there is no pLDDT or length filter to configure: every structure that parses and fits is tokenized, and the rest are rejected with a reason in `rejected/`. Residues missing any backbone atom (N, CA, C or O) get a `null` token, not a dropped position: the token list stays aligned to the sequence.
- Inference runs one structure at a time in FP32 by design: STōk found mixed-length batches alter terminal angles, so token identity is kept independent of batching and sharding. Don't expect batch-throughput scaling; expect shard-parallel scaling.
- Token identity is deterministic per device (the CLI turns on `torch.use_deterministic_algorithms`): reruns on the same device reproduce exactly, but CPU and GPU may differ near ties, and GPU model generations may differ from each other. `codebook_sha256` and `checkpoint_sha256` in each shard's footer pin which tokenizer produced it; compare those, not raw token values, when checking provenance.
- Only the Large preset is in this image. `--preset lite` would need its own checkpoint, which is not baked in: with the image offline it fails rather than downloading.
- The tool's test fixture is two 243-residue chains — well past the policy's 25-residue floor — one with a whole residue unobserved and one with a single missing O, so it exercises the null-token and null-coordinate paths, plus a 20-residue chain that must be rejected `chains_too_short`, and decodes both accepted rows' tokens back to within 1.5 Å RMSD of the input coordinates. Chains over 1280 residues (rejected `chains_too_long`) are the one policy branch it does not cover.
- The PyPI package named `stok` is an unrelated older project; this image installs STōk from GitHub at the pinned commit above.
- Weights are MIT licensed (both licenses sit beside the checkpoint in `/opt/stok-gcp`); STōk itself is MIT.
