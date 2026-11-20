---
name: igor
description: "IGoR, the V(D)J recombination model: aligns TCR and BCR reads to germline, infers a recombination model from them, scores each sequence's generation probability (Pgen) and most likely recombination scenarios, and generates synthetic repertoires. Reach for it for Pgen, recombination statistics, or sequences drawn from a fitted or published model."
metadata:
  version: 1.0.0
---
# igor

Image: `ghcr.io/brineylab/toolshed/igor:1.0.0`

IGoR 1.4.0 (Marcou, Mora & Walczak, *Nat. Commun.* 2018; github.com/qmarcou/IGoR), its last release, built from the release archive on Debian bookworm with a small patch (see Gotchas), run through its own `igor` program. Its published models are in the image at `/usr/local/share/igor/models`: human `alpha` (TRA), `beta` (TRB), `heavy_naive` (IGH), `kappa` (IGK) and `lambda` (IGL), and mouse `beta`. It needs no network. CPU only.

## Commands

Every command names a working directory (`-set_wd`), which must exist, and a batch name (`-batch`) that ties the steps of one dataset together; outputs land in subdirectories of the working directory. `-species` and `-chain` select a built-in model; `-set_custom_model <parms> <marginals>` uses any other, such as one you inferred. `-threads` sets the thread count (default: OpenMP's, every core the process may use, or `OMP_NUM_THREADS`).

Align reads to the model's germline genes, then score them: Pgen per sequence and each sequence's most likely scenarios.

```bash
igor -set_wd /work/igor -threads 8 -batch rep1 -species human -chain beta -read_seqs /work/reads.fasta -align --all
igor -set_wd /work/igor -threads 8 -batch rep1 -species human -chain beta -evaluate -output --Pgen --scenarios 10
```

Infer a recombination model from aligned reads (out-of-frame, unselected sequences are what IGoR's published models were learned from), then use it:

```bash
igor -set_wd /work/igor -threads 8 -batch rep1 -species human -chain beta -infer --N_iter 10
igor -set_wd /work/igor -threads 8 -batch rep1 -set_custom_model /work/igor/rep1_inference/final_parms.txt /work/igor/rep1_inference/final_marginals.txt -evaluate -output --Pgen
```

Generate synthetic sequences, with their CDR3s, from a model:

```bash
igor -set_wd /work/igor -threads 8 -batch sim -species human -chain heavy_naive -generate 10000 --noerr --CDR3 --seed 42
```

`--noerr` leaves out the model's sequencing-error (or, for IGH, hypermutation) process. The align step takes per-gene options (`--V`, `--D`, `--J`, each with `---thresh` and `---offset_bounds`); the full command reference is IGoR's documentation, qmarcou.github.io/IGoR.

## Inputs and outputs

- **In:** `-read_seqs` takes FASTA, a `.csv` of `seq_index;sequence` rows with a header, or a `.txt` with one sequence per line. Reads should be nucleotide sequences in the model's orientation, covering the CDR3 and enough V and J to align.
- **Out**, under the working directory:
  - `aligns/<batch>_{V,D,J}_alignments.csv` and `aligns/<batch>_indexed_sequences.csv` (each read's index), from `-align`.
  - `<batch>_output/Pgen_counts.csv` (`seq_index;Pgen_estimate`) and `<batch>_output/best_scenarios_counts.csv` (one row per scenario: `seq_index;scenario_rank;scenario_proba_cond_seq` then each event's realization as an index into the model, such as the V gene's position in `genomicVs.fasta`), from `-evaluate -output`.
  - `<batch>_inference/final_parms.txt` and `final_marginals.txt` (the inferred model), with per-iteration files and `likelihoods.out`, from `-infer`.
  - `<batch>_generated/generated_seqs_noerr.csv` (`seq_index;nt_sequence`), `generated_realizations_noerr.csv` (the scenario behind each), and with `--CDR3` `generated_seqs_noerr_CDR3_info.csv` (`seq_index,nt_CDR3,anchors_found,is_inframe`), from `-generate`. Without `--noerr` the names end in `_werr`.

## Compute

CPU only; memory grows with the number of reads and with the IGH model, which is much larger than the TCR ones. On a laptop at two threads, aligning 200 TRB sequences takes about 30 s, evaluating them a few seconds, and evaluating 40 IGH sequences about 10 s; generation is near instant. Alignment and evaluation scale linearly in reads and with threads: give a large repertoire a node's cores.

## Running it

On a laptop, from the directory holding `reads.fasta`:

```bash
mkdir -p igor
podman run --rm -v "$PWD":/work ghcr.io/brineylab/toolshed/igor:1.0.0 \
  igor -set_wd /work/igor -threads 8 -batch rep1 -species human -chain beta -read_seqs /work/reads.fasta -align --all
```

On a Slurm cluster with pyxis:

```bash
mkdir -p igor
srun --cpus-per-task=8 --container-image=ghcr.io#brineylab/toolshed/igor:1.0.0 --container-mounts="$PWD":/work \
  igor -set_wd /work/igor -threads 8 -batch rep1 -species human -chain beta -read_seqs /work/reads.fasta -align --all
```

## Gotchas

- IGoR 1.4.0 as released crashes when built by any GCC from 8 on: `Matrix::operator=` falls off its end without `return *this` (upstream issue #62, PR #67, never merged), which optimized builds turn into a segfault during generation and inference. This image's `igor-1.4.0.patch` adds that return and one more in `Deletion::has_effect_on`, which the build now treats as an error (`-Werror=return-type`); stops three FASTA and sequence readers from reading before the start of an empty line; and makes two changes it needs to compile (`#include <string>`, and `comp_nt_int` no longer `inline`). It changes no model or algorithm.
- `heavy_memory` is accepted and is the same model as `heavy_naive` in 1.4.0: its hypermutation branch is an empty TODO. The hypermutation models for IGH are in `/usr/local/share/igor/models/human/bcr_heavy/supplementary_models`, for use through `-set_custom_model`.
- The working directory must exist before the first command. If it does not, IGoR prints a `mkdir` error, writes nothing, and still exits 0: check that the output files exist, not only the exit code. A rerun with the same batch name overwrites its outputs.
- `-h` opens IGoR's man page, which the image does not carry; use the documentation.
- The test generates 200 TRB and 40 IGH sequences with seed 7, reads them back as FASTA with a blank line among them, aligns and evaluates them on two threads (IGoR's OpenMP path), and checks every sequence has Pgen > 0 under the model that generated it and that its best scenario names the V gene it was generated from; it then infers a TRB model in two iterations, evaluates the reads and generates under it, and generates from every built-in model. It does not run a repertoire-scale inference.
- IGoR and its models are GPLv3. The license, the release archive the binary was built from, and the patch are in the image at `/usr/local/share/igor/source`. A published model's provenance is its `model.info`, beside it under `/usr/local/share/igor/models`.
