---
name: igblast
description: "NCBI IgBLAST 1.22.0: annotates immunoglobulin rearrangements against IMGT human IG germline databases, with AIRR Rearrangement output. Reach for it to call V, D and J genes, the junction and CDR3, productivity and V identity on nucleotide antibody reads, the way the field's standard annotator is run."
metadata:
  version: 1.0.2
---
# igblast

Image: `ghcr.io/brineylab/toolshed/igblast:1.0.2`

NCBI IgBLAST 1.22.0 (the latest release, built October 2023; ncbi.nlm.nih.gov/igblast), its `igblastn` and `igblastp` beside `makeblastdb`, `blastdbcmd` and `edit_imgt_file.pl`, run the way its users run it: `-organism human -ig_seqtype Ig` against IMGT human IG germline V, D and J databases (one V, one D, one J database over IGH, IGK and IGL together), prepared exactly as NCBI's setup guide says — `edit_imgt_file.pl`, then `makeblastdb -parse_seqids` — with NCBI's patched `optional_file/human_gl.aux` (2026-06-16) and the release's `internal_data`. Everything lives under `/usr/local/share/igblast`, and the programs are also linked into `/usr/local/bin`, so `igblastn` is on PATH however the image is run; `IGDATA` points there, so `internal_data` and `optional_file` are found without options. It needs no network. CPU only.

## Commands

Annotate nucleotide reads, AIRR Rearrangement TSV out (every other option default):

```bash
igblastn -germline_db_V /usr/local/share/igblast/database/imgt_human_ig/imgt_human_ig_v.fa \
  -germline_db_D /usr/local/share/igblast/database/imgt_human_ig/imgt_human_ig_d.fa \
  -germline_db_J /usr/local/share/igblast/database/imgt_human_ig/imgt_human_ig_j.fa \
  -organism human -ig_seqtype Ig \
  -auxiliary_data /usr/local/share/igblast/optional_file/human_gl.aux \
  -query /work/seqs.fa -out /work/seqs_airr.tsv -outfmt 19 -num_threads 8
```

The same run in IgBLAST's own text layout (`-outfmt 3`; `2` and `7` are variants, `blast`-like and tab-separated): one block per query, opening with its V-(D)-J rearrangement summary (top V, D and J genes, chain type, productivity), then junction and sub-region details, then the per-region alignment summary, and ending in BLAST-like pairwise alignments:

```bash
igblastn -germline_db_V /usr/local/share/igblast/database/imgt_human_ig/imgt_human_ig_v.fa \
  -germline_db_D /usr/local/share/igblast/database/imgt_human_ig/imgt_human_ig_d.fa \
  -germline_db_J /usr/local/share/igblast/database/imgt_human_ig/imgt_human_ig_j.fa \
  -organism human -ig_seqtype Ig \
  -auxiliary_data /usr/local/share/igblast/optional_file/human_gl.aux \
  -query /work/seqs.fa -out /work/seqs.tsv -outfmt 3 -num_threads 8
```

Build a germline database of your own from an IMGT-format FASTA the way the image's were built (`-parse_seqids` matters: IgBLAST reports gene names from the database's seqids):

```bash
edit_imgt_file.pl /work/germline_v.fasta > /work/my_v.fa
makeblastdb -parse_seqids -dbtype nucl -in /work/my_v.fa
```

## Inputs and outputs

- **In:** nucleotide FASTA (or FASTQ) of antibody rearrangements, the first word of each header the sequence's id; `-germline_db_V/D/J` accepts any BLAST nucleotide database. `-ig_seqtype Tcr` with `-germline_db_*` of TR genes annotates TCRs; `-organism` also knows mouse, rat, rabbit and rhesus_monkey (with their `internal_data`), or any name, given custom internal data as IgBLAST's documentation describes.
- **Out, `-outfmt 19`:** one row per query in the AIRR Rearrangement TSV (v_call, d_call, j_call, junction, junction_aa, cdr3, cdr3_aa, np1, np2, fwr1–fwr4, cdr1–cdr2, v_identity, d_identity, j_identity, productive, stop_codon, vj_in_frame, v_frameshift, d_frame, rev_comp, complete_vdj, vj_in_frame, and the alignment coordinates). **`-outfmt 3/2/7/blast`:** IgBLAST's own per-query annotation block (V-(D)-J rearrangement summary, junction and sub-region details, per-region alignment summary) followed by BLAST-like pairwise alignments. Warnings (queries with no V hit, J genes missing from the aux file) go to stderr; every query still gets an AIRR row.

## Compute

CPU only; `-num_threads` is the knob, default 1 (the BLAST default), and the databases are a few MB. On one core of a modest shared CPU it annotates about 25 full-length heavy chains a second; a 2.4-million-chain repertoire is an hour on 32 cores. Memory is small and flat in the number of queries.

## Running it

On a laptop, from the directory holding `seqs.fa`:

```bash
podman run --rm -v "$PWD":/work ghcr.io/brineylab/toolshed/igblast:1.0.2 \
  igblastn -germline_db_V /usr/local/share/igblast/database/imgt_human_ig/imgt_human_ig_v.fa \
  -germline_db_D /usr/local/share/igblast/database/imgt_human_ig/imgt_human_ig_d.fa \
  -germline_db_J /usr/local/share/igblast/database/imgt_human_ig/imgt_human_ig_j.fa \
  -organism human -ig_seqtype Ig \
  -auxiliary_data /usr/local/share/igblast/optional_file/human_gl.aux \
  -query /work/seqs.fa -out /work/seqs_airr.tsv -outfmt 19 -num_threads 8
```

On a Slurm cluster with pyxis (a CPU partition):

```bash
srun --cpus-per-task=8 --mem=8G --container-image=ghcr.io#brineylab/toolshed/igblast:1.0.2 \
  --container-mounts="$PWD":/work --container-workdir=/work \
  igblastn -germline_db_V /usr/local/share/igblast/database/imgt_human_ig/imgt_human_ig_v.fa \
  -germline_db_D /usr/local/share/igblast/database/imgt_human_ig/imgt_human_ig_d.fa \
  -germline_db_J /usr/local/share/igblast/database/imgt_human_ig/imgt_human_ig_j.fa \
  -organism human -ig_seqtype Ig \
  -auxiliary_data /usr/local/share/igblast/optional_file/human_gl.aux \
  -query /work/seqs.fa -out /work/seqs_airr.tsv -outfmt 19 -num_threads 8
```

## Gotchas

- **Under pyxis the image's `ENV PATH` does not reach the job** — enroot runs the container's commands on the distro's default PATH, which `/usr/local/share/igblast/bin` is not on — so the programs are symlinked into `/usr/local/bin` (1.0.0's test failed on exactly this: `igblastn: command not found`). `IGDATA` and `LC_ALL` from the image's ENV do survive pyxis, so `internal_data` is found and sorting is byte-wise.
- **IMGT blocks some networks.** The image's germline databases were fetched at build time from IMGT's V-QUEST reference directory when it answered, and otherwise from the identical files vendored in the partis repository's tree (its `data/germlines/imgt-download/`, IMGT's own FASTAs, unmodified); `database/imgt_human_ig/PROVENANCE.txt` records, per file, which source was used, its sha256 and the fetch date. IMGT does not version the individual files; the reference directory's own release at fetch time is published on IMGT's V-QUEST reference page (release 202643-4, 22 October 2026, when this was written).
- `v_call` and `j_call` can hold several equally best alleles, comma-separated (the fixture's IGHV3-23 chains also hit its duplicate IGHV3-23D); read those fields as a list. Light chains get an empty `d_call`.
- IgBLAST's junction and CDR3 run from the conserved cysteine (IMGT position 104) to the conserved tryptophan (position 118), inclusive, which is one codon longer at each end than IMGT's CDR3 definition; `junction_aa_length` and `cdr3_aa_length` follow it.
- `-auxiliary_data` is passed explicitly in every example, but with `IGDATA` set it is also the default; the file must match the J genes in `-germline_db_J` (the image's patched human_gl.aux covers every J allele in the IMGT human IG files it shipped with; a J gene missing from it costs its CDR3 stop, and with it junction, CDR3 and productivity, but not the gene calls).
- The IMGT files hold functional, ORF and in-frame pseudogene alleles, with IMGT's gaps removed by `edit_imgt_file.pl`; orphons (names ending in D, -2, or P) are kept, as `edit_imgt_file.pl` keeps them.
- IgBLAST is public domain, a United States government work (the release's LICENSE is in the image at /usr/local/share/igblast/LICENSE). IMGT® germline sequences are free for research and academic use only, with commercial use requiring a license from IMGT; cite Lefranc et al., the international ImMunoGeneTics information system.
- The test's fixture is fourteen productive rearrangements joined out of IMGT germline V, (D) and J regions by `make_fixture.py` (seeded; eight carry 3-6 somatic substitutions at positions drawn from across V, framework and CDR alike, v_identity 97.95 to 100%), so the V, D and J calls are known by construction; it exercises all three loci, D reading frame, gene-name ties, and the stop-codon and frame logic of the aux file. igblastn has no input-size threshold; the fixture covers single- and multi-allele calls, not scale.
