---
name: partis
description: "partis, the BCR/TCR repertoire toolkit: annotates sequences (AIRR output), infers germline and repertoire parameters, partitions sequences into clonal families with heavy/light pairing, computes selection metrics, and simulates realistic repertoires with per-base mutation along TreeSim trees. Reach for it to annotate or cluster a repertoire, or to simulate one with a known truth."
metadata:
  version: 1.0.0
---
# partis

Image: `ghcr.io/brineylab/toolshed/partis:1.0.0`

partis v1.0.30 (Ralph & Matsen; github.com/psathyrella/partis), run from its source tree at the tag's commit `40ff8e3`, the way upstream's own image runs it: `bcrham` and `ig-sw` compiled in place, and the simulation stack built too, which the pip package leaves out: bpp-newlik's `bppseqgen` (Bio++, built with `bin/build.sh with-simulation`) for the default per-base mutation model, and R 4.5.3 with TreeSim, TreeSimGM, geiger and MASS (from CRAN's snapshot of 2026-04-23) for trees. Its germline sets (human, macaque, mouse strains and more, IG and TR loci, from IMGT and, for macaque, OGRDB) and its pooled mutation tables for simulating from scratch ship in `/opt/partis/data`. It needs no network. CPU only.

## Commands

Annotate sequences, inferring parameters (germline set, deletions, insertions, mutation) from them first; `--airr-output` writes an AIRR TSV beside the YAML:

```bash
partis annotate --infname /work/seqs.fa --parameter-dir /work/params --outfname /work/ann.yaml --airr-output --n-procs 8
```

Partition a single-chain sample into clonal families, or a heavy/light sample using its pairing (from 10x droplet ids in the sequence names, or `paired-uids` in `--input-metafnames`):

```bash
partis partition --infname /work/seqs.fa --parameter-dir /work/params --outfname /work/partition.yaml --n-procs 8
partis partition --paired-loci --infname /work/all-seqs.fa --input-metafnames /work/meta.yaml --paired-outdir /work/ppart --n-procs 8
```

A paired run writes `partition-igh.yaml`, `partition-igk.yaml` and `partition-igl.yaml` (joint partitions) and the per-locus parameters under `--paired-outdir`. A directory laid out that way, such as a paired simulation's, is read with `--paired-indir` in place of `--infname`.

Selection metrics (aa-cdist, lbi, lbr) on a partition, then parse partis output into FASTA, CSV, or AIRR:

```bash
partis get-selection-metrics --paired-loci --paired-outdir /work/ppart --n-procs 8
parse-output.py /work/ppart/partition-igh.yaml /work/igh.fa
parse-output.py /work/ppart/partition-igh.yaml /work/igh.tsv --airr-output
```

Simulate. From scratch, with realistic per-position, per-base mutation from partis's pooled public-data tables; mimicking a sample, from the parameter directory inferred from it; or paired heavy/light:

```bash
partis simulate --rearrange-from-scratch --outfname /work/simu.yaml --n-sim-events 1000 --seed 1 --n-procs 8
partis simulate --parameter-dir /work/params --outfname /work/simu.yaml --n-sim-events 1000 --seed 1 --n-procs 8
partis simulate --paired-loci --rearrange-from-scratch --paired-outdir /work/psimu --n-sim-events 1000 --seed 1 --n-procs 8
```

Each event is a clonal family: a rearrangement, a tree from TreeSim, and mutation along it by `bppseqgen`. `--mutation-multiplier`, `--n-leaf-distribution`, `--n-leaves`, and `--light-chain-fractions` shape it; `--no-per-base-mutation` swaps in the older, ten times faster, per-position-only model. The output YAML carries the truth (`reco_id` per family, the true annotation per sequence); run inference on it with `--is-simu` and partis prints its accuracy, purity and completeness for a partition. `partis <action> --help` lists every option; the documentation is partis's `docs/` on GitHub.

## Inputs and outputs

- **In:** FASTA (or partis YAML, or AIRR TSV with `--airr-input`) of positive-sense sequences of one locus, `--locus igh` by default (`igk`, `igl`, `tra`, `trb`, ...) and `--species human` by default; `--paired-loci` takes all loci mixed.
- **Out:** partis YAML (`--outfname`), holding the germline set, annotations, and for `partition` the clustering path, read in Python with `partis.utils.read_output(fname)` under the image's python (`/opt/venv/bin/python3 -I`, where `partis` imports); AIRR TSV with `--airr-output`; `selection-metrics.yaml` in the paired outdir; parameters (`sw/`, `hmm/`, and for simulations `true/`) under `--parameter-dir`.

## Compute

CPU only; `--n-procs` defaults to every core. Upstream's figures: annotating 10,000 sequences takes about 5 minutes on 8 cores and partitioning them about 25; simulating 10,000 sequences with per-base mutation about 8 minutes on one core, ten times less with `--no-per-base-mutation`. On a laptop at two processes, this tool's test simulates 20 IGH families (86 sequences) in about 5 s and annotates them in 5 s, and partitions 40 simulated paired families (about 430 sequences) in about 30 s; the whole test takes about 70 s. Temporary files go to `--workdir`, by default under `/tmp`; on a cluster running partis's own Slurm parallelism (`--batch-system slurm`), set it to a shared path.

## Running it

On a laptop, from the directory holding `seqs.fa`:

```bash
podman run --rm -v "$PWD":/work ghcr.io/brineylab/toolshed/partis:1.0.0 \
  partis annotate --infname /work/seqs.fa --parameter-dir /work/params --outfname /work/ann.yaml --airr-output --n-procs 8
```

On a Slurm cluster with pyxis:

```bash
srun --cpus-per-task=8 --mem=16G --container-image=ghcr.io#brineylab/toolshed/partis:1.0.0 --container-mounts="$PWD":/work \
  partis simulate --rearrange-from-scratch --outfname /work/simu.yaml --n-sim-events 1000 --seed 1 --n-procs 8
```

## Gotchas

- Upstream's own CI never exercises the default per-base simulation or TreeSim tree generation (it tests `--no-per-base-mutation --no-tree-gen`); this tool's test is what checks that path. With `--debug 1`, a simulation logs `generating N trees` and, for every family, `using bpp per-base mutation with path /opt/partis/packages/bpp-newlik/_build`; without R or bpp-newlik it fails rather than falling back.
- With `--is-simu`, a paired partition raises `Exception: missing 0.089 (more than 0.050)` when more than 5% of a light chain's true sequences are missing from its partition (they failed annotation): a deliberate guard in partis's comparison against the truth, not a failed partition. On 40 small simulated families it tripped in about 1 run in 10 (5 of 56 IGL sequences). The test scores partitions itself and asks for at least 85% coverage instead. The partition is not deterministic run to run.
- `colored_traceback`, which every partis program imports only to color tracebacks, kills the program under any `TERM` the image lacks terminfo for, and pyxis passes the caller's `TERM` in; the image replaces it with an empty module, so tracebacks print plain.
- partis's version reads as `unknown`: the image keeps the source tree but not its `.git`. It is v1.0.30, commit `40ff8e3c9b07309193aa76c29652ca4a2531f101`.
- Simulating from scratch reads per-base mutation counts for genes that have none from their nearest relatives and warns that the counts will be wrong where the genes differ; that is partis's documented behavior for `--rearrange-from-scratch`.
- The test runs every command above on two processes (partis's multiprocess path). It simulates 20 IGH families with per-base mutation and TreeSim trees and checks every family went through bpp-newlik; parses them to FASTA, annotates them (the true V gene for at least 90% of the 86 sequences, 82 here, and at least 80% mutated in V) and partitions them (purity at least 0.95, completeness at least 0.75 against the true families); simulates 10 more families from the inferred parameters; simulates 40 paired families and partitions their mixed FASTA with its pairing, requiring per chain at least 85% of sequences partitioned, purity at least 0.95, completeness at least 0.75, and no chain less pure paired than alone (IGK went from 0.815 alone to 1.000 paired in one run); then computes selection metrics (six families reach the tree-size minimum) and parses the IGH partition to FASTA and AIRR, one row per sequence. It does not run a repertoire-scale partition.
- partis is GPLv3, Bio++ CeCILL, TreeSim and TreeSimGM GPL-2; the full partis source, with its submodules' sources and licenses, is in the image at `/opt/partis`.
