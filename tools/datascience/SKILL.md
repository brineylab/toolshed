---
name: datascience
description: "General-purpose data work: the scientific Python stack, abstar and abutils, ANARCII, mmseqs2, and DIAMOND, with network access. Reach for it to annotate, number, search, or reshape sequence data, and to run a brineylab/datasets recipe."
metadata:
  version: 1.2.0
---
# datascience

Image: `ghcr.io/brineylab/toolshed/datascience:1.2.0`

Python 3.12 with numpy, polars, pandas, pyarrow, duckdb, scipy, scikit-learn, biopython, matplotlib, seaborn, `huggingface_hub`, and `datasets`; abstar 0.9.0 and abutils 0.6.1 with their germline databases; ANARCII 2.0.8 (IMGT numbering of amino-acid antibody sequences; its model weights ship in the package) on CPU-only torch 2.15.0; MMseqs2 built from master (commit 2a3dafc), past release 18-8cc5c; DIAMOND 2.2.8; and git. It reaches the network: a script in it may download its inputs and push its outputs.

## Commands

```bash
abstar run seqs.fasta abstar_out --germline_database human -o parquet
python3 -I -c "from anarcii import Anarcii; print(Anarcii(seq_type='antibody', cpu=True).number({'q': 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYAMYWVRQAPGQRLEWMGWINARNGNTKYSQKFQGRVTITRDTSASTAYMELSSLRSEDTAVYYCARSPYSSSWPGYYYYYYGMDVWGQGTTVTVSS'})['q']['chain_type'])"
mmseqs easy-cluster seqs.fasta clu tmp --min-seq-id 0.85 -c 0.8
diamond blastp -d db.dmnd -q seqs.fasta -o hits.tsv
python3 script.py
```

`abstar run` takes a FASTA or a directory of them; `mmseqs` and `diamond` take their usual subcommands (`mmseqs createdb`, `diamond makedb`).

## Inputs and outputs

- **abstar:** nucleotide FASTA of antibody sequences, the first word of each header the sequence's id. Out: parquet files under the project directory, one row per sequence, with `sequence_id`, `locus`, `productive`, the V, D, and J calls, the regions `fwr1_aa` to `fwr4_aa` and `cdr1_aa` to `cdr3_aa`, and `sequence_vdj_aa`, the variable domain from FWR1 through FWR4 (the joined regions on a productive row).
- **ANARCII:** `Anarcii(seq_type="antibody", mode="accuracy", cpu=True, ncpu=<n>).number({id: amino_acids})` returns `{id: {"numbering": [((position, insertion), residue), ...], "chain_type": "H" | "K" | "L" | "F", ...}}`: IMGT positions with insertions as letters (`(111, "A")`), gaps as `"-"`, and `F` with an `error` when it cannot number a sequence. IMGT CDR3 is positions 105 to 117.
- **mmseqs `easy-cluster`:** a FASTA. Out: `<prefix>_cluster.tsv` (representative, member), `<prefix>_rep_seq.fasta`, `<prefix>_all_seqs.fasta`.
- **diamond:** protein or translated searches against a `.dmnd` database built with `diamond makedb --in ref.fasta -d db`. Out: BLAST tabular (`-f 6`) by default.

## Compute

CPU only. The test (600 contigs through abstar, their 599 productive variable domains numbered with ANARCII, then the heavy chains clustered with mmseqs and searched against themselves with DIAMOND) takes about 20 s on a laptop. abstar annotated the fixture at about 110 sequences a second on a laptop and 75 on 4 CPUs of an hpc-low node, in chunks of 500 sequences across a process pool; mmseqs and diamond take `--threads`.

## Running it

On a laptop, from the directory holding the inputs:

```bash
podman run --rm -v "$PWD":/work -w /work ghcr.io/brineylab/toolshed/datascience:1.2.0 \
  abstar run seqs.fasta abstar_out --germline_database human -o parquet
```

On a Slurm cluster with pyxis (a CPU partition; on coreweave-rtx6000, `hpc-low`):

```bash
srun --partition=hpc-low --cpus-per-task=8 --mem=32G \
  --container-image=ghcr.io#brineylab/toolshed/datascience:1.2.0 \
  --container-mounts="$PWD":/work --container-workdir=/work \
  abstar run seqs.fasta abstar_out --germline_database human -o parquet
```

## Gotchas

- A general-purpose image reaches the network and sets no `HF_HUB_OFFLINE`. Pyxis passes the caller's environment in, so a caller's `HF_HUB_OFFLINE=1` stops a script's downloads and its push to the Hub: unset it first.
- `mmseqs` is MMseqs2 built from master (commit 2a3dafc) for the fix to #1092, not abutils' bundled build (commit 7e28409), which abstar still runs from its own path: the bundled build dies in `easy-cluster`'s prefilter on about 1.1M sequences of 240 residues, as does release 18. Abstar's germline searches are small and unaffected.
- ANARCII runs on CPU here (`cpu=True`); there is no CUDA in this image. Past 102,400 sequences in one `number` call it writes its results to a msgpack file in the working directory instead of returning them: number large sets in calls below that.
- git trusts every directory (`safe.directory '*'`): a checkout owned by another user, as under pyxis or podman, still answers `git status`.
- The test's fixture is 600 contigs from Jaffe et al. 2022 (figshare 19617633 v3, MIT), 300 droplets of sample 1279049 with one heavy and one light chain: it crosses abstar's 500-sequence chunk into its process pool, and does not reach `mmseqs_chunksize` (1,000,000), where abstar splits mmseqs' input. ANARCII and abstar agree on all 599 productive chains' locus and CDR3, 216 of them CDR3s longer than 13 residues, which carry IMGT insertions.
