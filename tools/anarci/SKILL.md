---
name: anarci
description: "Classic ANARCI/HMMER IMGT numbering with CSV output for antibody and receptor sequences. Reach for it when a reproducible analysis requires ANARCI 2026.2.13.2's HMM alignments and exact CSV numbering, including germline assignment, rather than ANARCII's learned model."
metadata:
  version: 1.0.1
---
# anarci

Image: `ghcr.io/brineylab/toolshed/anarci:1.0.1`

ANARCI 2026.2.13.2 with its packaged HMMs and germline tables, HMMER 3.3.2 (`hmmscan`/`hmmpress`), Biopython 1.88, and the scientific Python/parquet stack. ANARCI classifies and numbers variable domains under IMGT, Kabat, Chothia, Martin, Wolfguy, or AHo. The image is CPU-only and self-contained; no network or model download is needed at run time.

## Commands

Number translated human heavy-chain sequences exactly as EPAM's `scripts/run_anarci_epam_naive_v2.sh` did, with one process per numbered chunk:

```bash
ANARCI -i /work/parents.fasta -o /work/parents_imgt \
  -s imgt -r heavy --use_species human --csv -p 5 --assign_germline
```

For light chains, use `-r light`. For a quick known-answer check of one sequence, omit `--csv` and read the vertical ANARCI output:

```bash
ANARCI -i QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYAMYWVRQAPGQRLEWMGWINARNGNTKYSQKFQGRVTITRDTSASTAYMELSSLRSEDTAVYYCARSPYSSSWPGYYYYYYGMDVWGQGTTVTVSS -s imgt -r heavy --use_species human
```

An input `parents.fasta` may use EPAM's three-part ids:

```text
>{sample_id}|{family}|{seq_name}
<translated parent amino acids>
```

## Inputs and outputs

- **Input:** one protein sequence string or a multi-record amino-acid FASTA. Use ungapped translations of the parent sequences. The FASTA id becomes ANARCI's CSV `Id`; without whitespace in it, ids such as `sample|family|name` round-trip exactly.
- **CSV output:** `-o /work/parents_imgt --csv` writes `/work/parents_imgt_H.csv` for heavy hits, `..._KL.csv` for kappa/lambda, and analogous files for recognized receptor types. A record with no passing domain is omitted, not represented by a null row. Multiple domains have `domain_no` 0, 1, etc.; EPAM parents are expected to have a single heavy domain.
- The CSV's first 13 fields are `Id`, `domain_no`, `hmm_species`, `chain_type`, `e-value`, `score`, `seqstart_index`, `seqend_index`, `identity_species`, `v_gene`, `v_identity`, `j_gene`, `j_identity`. Every later header is an IMGT position with an optional insertion letter. Each cell is the aligned amino acid, or `-` where that sequence has no residue at a position.
- Keep the CSV columns in their emitted order. ANARCI ranks repeated integer positions by encounter order, preserving IMGT's `111`/`112` insertion convention (CDR3 insertions extend `111-ABC CBA-112`). Do not sort numeric labels and insertion labels separately or collapse distinct columns into a key/value map.
- `--use_species human --assign_germline` restricts V/J identity assignment to the human germline tables and populates `identity_species`, V/J gene and identity fields. `-p` is ANARCI's process-pool size; HMMER is forced to one thread per process.

## Compute

CPU only. `-p N` uses N processes and HMMER uses one thread in each; memory is roughly proportional to input size because ANARCI reads the FASTA and compiled results in memory. Split very large corpora into resumable chunks and use one `-p` allocation that fits the job. The four-record no-network test is interactive in latency; it checks process-pool operation and CSV correctness, not corpus throughput.

## Running it

On a laptop, from a directory containing `parents.fasta`:

```bash
podman run --rm --network none -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/anarci:1.0.1 \
  ANARCI -i /work/parents.fasta -o /work/parents_imgt \
  -s imgt -r heavy --use_species human --csv -p 5 --assign_germline
```

On a Slurm cluster with pyxis:

```bash
srun --partition=hpc-low --cpus-per-task=5 --mem=16G \
  --container-image=ghcr.io#brineylab/toolshed/anarci:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work \
  ANARCI -i /work/parents.fasta -o /work/parents_imgt \
  -s imgt -r heavy --use_species human --csv -p 5 --assign_germline
```

## Gotchas

- ANARCI is the original HMM/HMMER software, distinct from the ANARCII model published in `datascience`. Both output IMGT, but they are not drop-in numerically equivalent implementations.
- A missing sequence in the chain-specific CSV means no domain passed, or the requested `-r` restriction rejected it. Do not silently interpret that as successful complete-domain numbering. Inputs shorter than a recognizable domain, non-heavy sequences, corrupted sequences and multiple-domain hits need explicit handling by the caller.
- CSV `seqstart_index`/`seqend_index` locate recognized domains. Reconstructing the input by deleting `-` cells checks complete alignment coverage but does not by itself validate EPAM's CDR annotation; downstream consumers must compare CDR coordinates independently and can reject a family on mismatch.
- ANARCI and MUSCLE's upstream licenses ship with the wheel. ANARCI is BSD-3-Clause; its bundled HMM/germline resources have no separately asserted model-weights license and are redistributed as part of that permissively licensed software package.
- The four-record fixture contains three translated parents from EPAM `ece4e972`'s public `data/parent-child-example.csv` and one insulin-like negative control. It crosses ANARCI's multi-process boundary (`-p 2`), requires human germline assignment, covers IMGT `111`/`112` insertions and exact input reconstruction. It does not assert real-FLAIRR exclusion tallies or large-corpus scaling.
- `HF_HUB_OFFLINE=1` is the shipped default because no numbering resource needs the network. A dataset recipe that downloads its pinned source must override that variable with `HF_HUB_OFFLINE=0` for its download step; do not bake a Hub credential into the image.
