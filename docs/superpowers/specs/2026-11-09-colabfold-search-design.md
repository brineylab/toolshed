# colabfold-search and colabfold-db: Design

Status: draft 2026-11-09, from a brainstorm with the human. Makes amendments, approved with this design, to the toolshed's `docs/DESIGN.md` and to the datasets repository's `docs/DESIGN.md` (§6); both are approved by the lab and change only with its approval. Calls made in writing that nobody has discussed are marked **(call)**.

## 1. Why

Two pieces of work need ColabFold MSAs now, more than the public ColabFold server will serve (its rules allow serial queries from one machine):

1. **Structure prediction at scale**: many Boltz-2 runs, which today either run single-sequence (`msa: empty`) or call the public server with `--use_msa_server`.
2. **A coevolution metric**: a few thousand MSAs for a new precision-at-L (P@L) eval of contacts. The lab has no convention for these MSAs yet; what it wants is the most accurate one.

In both, every sequence is known before any prediction starts. So both are batch searches: a FASTA in, one `.a3m` per sequence out, files in a mounted directory. That is the shape of every tool in the shed. What is new is the database: about 240 GB compressed and several hundred GB unpacked, far too big for an image.

## 2. Goals, non-goals, anti-goals

**Goals.**
- A tool, `colabfold-search`, that runs ColabFold's own search (`colabfold_search`) on CPU against a local copy of ColabFold's databases, with two documented command lines: one that matches what the public server returns, for structure prediction, and one that returns the deepest, unfiltered MSA, for coevolution.
- A dataset, `colabfold-db`, in `brineylab/datasets`: ColabFold's two prebuilt MMseqs2 databases, mirrored byte for byte on the Hub, so that a name and version are always the same bytes.
- A documented, once-per-cluster step that puts an unpacked copy of a `colabfold-db` version on the cluster's shared storage, where search jobs mount it read-only.
- First cluster: coreweave-rtx6000.

**Non-goals (deferred, each with its trigger, §7).**
- An MSA *service* on a port. Not needed: every sequence is known up front.
- GPU search (MMseqs2-GPU).
- Paired MSAs for complexes, and templates.
- A copy on `bcr`.

**Anti-goals: what would make this a failure even if it ran.**
- A result whose MSA cannot be traced to exact database bytes and exact tool version.
- MSA filtering conventions for the coevolution metric baked into the tool, where they would be invisible to the metric and fixed per tool version. The metric owns its filtering (§4).
- A tool that reaches the network at run time, or a job that downloads 240 GB as a side effect.
- CPU and GPU MSAs mixed silently in one evaluation (their prefilters differ, so their MSAs differ).

## 3. The dataset: `colabfold-db`

A dataset in `brineylab/datasets`, `datasets/colabfold-db/`, following that repository's rules except where §6 amends them.

**Files.** Exactly the upstream archives, unchanged, as the version's files:

| File | Upstream | Size (bytes, upstream `Content-Length` on 2026-11-09) |
|---|---|---|
| `uniref30_2302.db.tar.gz` | `https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz` | 106,729,528,496 |
| `colabfold_envdb_202108.db.tar.gz` | `https://opendata.mmseqs.org/colabfold/colabfold_envdb_202108.db.tar.gz` | 128,576,661,224 |

Each is under the Hub's 200 GB per-file guidance; together about 236 GB. These are ColabFold's "fast prebuilt" databases (`FAST_PREBUILT_DATABASES=1` in its `setup_databases.sh`), which serve CPU and GPU search alike. Upstream re-uploaded both under the same names in April 2025 (`Last-Modified`), so a name upstream does not fix its bytes; the mirror does. Left out **(call)**: `uniref30_2302_newtaxonomy.tar.gz` (only for paired MSAs), the PDB100 and Foldseek template files (only for templates), and `spire_ctg10_2401` (environmental pairing). Each comes in a later version when its deferred item does.

**Recipe** (`recipe/build <out>`, in `datascience@1.1.0`, one CPU job with network). Downloads each file into `<out>/` (not `.work/`: the archives *are* the version's files), streaming and hashing as it goes, as `jaffe-paired`'s recipe does with figshare. `recipe/params.toml` holds the pull date, the two URLs, and each file's expected size and sha256. Upstream publishes no checksums, so the first cook's download fixes the sha256s **(call)**: the cook downloads once, writes the sha256s into `params.toml`, commits, and builds again from the committed recipe into the same output (the recipe skips a file already present whose size and sha256 match, and refuses one that differs). A later re-pull whose bytes differ is a new version.

**Test** (`recipe/test <out>`, in seconds as the rules require). Checks that the two files exist with the sizes in `params.toml` and that each is a gzip tar whose first members are the MMseqs2 database files expected (`uniref30_2302_db*`, `colabfold_envdb_202108_db*`), reading only the first blocks. It does not hash 236 GB: the recipe verified the hashes on download and `scripts/push` writes them into the record's manifest.

**Skill frontmatter.** `name: colabfold-db`, `metadata.tools: datascience@1.1.0`, `metadata.license`: the source's license, which the builder confirms from ColabFold's documentation before pushing (CC BY-SA 4.0 is expected for both; the Hub license id is `cc-by-sa-4.0`). The skill's prose says what the databases are, the upstream URLs and pull date, that the files are archives to unpack (not to read), and points to `colabfold-search`'s skill for unpacking and use.

**Push.** `scripts/push` uploads with `upload_folder` in one commit. 236 GB in two files is far beyond what any dataset so far has pushed. The builder confirms it completes (and resumes if interrupted) before relying on it; if it does not, switching `push` to `upload_large_folder` is a change to the datasets repository's scripts and its unit tests, made in its own pull request first. The lab has confirmed its Hub org holds a ~240 GB public dataset (the human, 2026-11-09).

**Compute.** One CPU job on coreweave-rtx6000 `hpc-low`, a few CPUs and little memory, network, about 240 GB of disk for `<out>` plus the same again on the Hub cache once pulled. Time is the download and upload; the skill records what it took.

## 4. The tool: `colabfold-search`

`tools/colabfold-search/` in the shed, following every rule in `AGENTS.md`.

**Image.**
- A glibc Python base (`python:3.12-slim-bookworm` pinned by digest, as `datascience`), `colabfold` pinned (latest release on 2026-11-09 is `v1.6.3`), installed without the `alphafold`/JAX extras if `colabfold_search` runs without them (the builder checks).
- MMseqs2 **Release 18** (`18-8cc5c`), the static CPU (AVX2) binary from its GitHub release, sha256-checked, on PATH as `mmseqs`. ColabFold says Release 18 gives the same MSAs as its server. **(call)** The CPU binary, not the GPU one: GPU search is deferred, and adding it is a new version anyway.
- No weights and no database. `HF_HUB_OFFLINE=1` is set as the rules require of a non-general-purpose tool, though nothing here uses the Hub. No NVIDIA variables: it is not a GPU tool until GPU search arrives.
- `colabfold_search` is a Python console script; the image must keep the caller's environment and home out of it (pyxis passes both in), in the spirit of the `python3 -I` rule: the builder makes the installed entry point run isolated (for example a shebang with `-I`, or `PYTHONNOUSERSITE=1` and no `PYTHONPATH` honoured) and the test checks it with a hostile `PYTHONPATH` set.
- Ends with `COPY test /toolshed/test`.

**The database inside a job.** The caller mounts an unpacked `colabfold-db` version read-only at `/db` (§5). `colabfold_search`'s default database names (`uniref30_2302_db`, `colabfold_envdb_202108_db`) are the dataset's, so no `--db1`/`--db3` flags are needed. The skill names the `colabfold-db` version this tool version is documented against; a result is identified by both versions.

**Commands** (both CPU; the input is a FASTA with one chain per record and no `:`-joined complexes):

- **For structure prediction** (what the public server returns, unpaired):
  `colabfold_search --threads 32 chains.fasta /db msas`
  ColabFold's defaults: `--filter 1` (alignment cutoffs fixed at `qsc 0.8`, `max-accept 100000`; the final MSA reduced to sequences under 95% identity to each other and about 3,000 per identity band), UniRef30 and the environmental database both searched. Each chain gets its own MSA; a Boltz-2 YAML points each protein chain's `msa:` at its `.a3m`.
- **For coevolution** (deepest, unfiltered):
  `colabfold_search --threads 32 --filter 0 seqs.fasta /db msas`
  No diversity reduction, `qsc -20`, up to `max-accept 1000000` alignments per query per database. Filtering and weighting are the caller's: the P@L code drops low-coverage rows and reweights near-duplicates as part of the metric's definition (standard practice is coverage ≥ 75% and weights at 80% identity, but that is the metric's call, not the tool's). Keeping filtering out of the tool lets one MSA set be rescored under a changed convention without a new search.

`colabfold_search` writes its intermediates in the output directory and needs room; the skill says how much once measured. Sensitivity (`-s`) stays at ColabFold's default in both lines **(call)**: the first real coevolution batch reports Neff per MSA, and if depth limits P@L, a raised `-s` is a later version.

**Inputs and outputs.** In: a FASTA. Out: `msas/<name>.a3m`, one per record, query first, in A3M. The builder confirms from a run how `colabfold_search` names its outputs (from the FASTA header, or by index) and the skill states it, with how headers map to files.

**Test** (`test/run.sh`, CPU, `--network none`, writes only under `/tmp`). The fixture is kilobytes: a query and a handful of planted homologs. The test builds a tiny ColabFold-shaped database in `/tmp` from the fixture with `mmseqs` (the layout `colabfold_search` expects of `uniref30_2302_db` and `colabfold_envdb_202108_db`: the builder works this out from `setup_databases.sh` and `colabfold/mmseqs/search.py`; ColabFold's own `mini_swissprot2503` debug database is 276 MB, too large for a fixture), then runs **both documented command lines** against it and checks the known answer: the planted homologs appear in the query's `.a3m`, and the `--filter 0` MSA holds at least as many rows as the default one. It also runs `colabfold_search` with a `PYTHONPATH` pointing at a directory holding a broken module of a name ColabFold imports, and passes only if the search is unaffected.

**Compute.** CPU only. ColabFold's README asks about 128 GB of RAM for batch search. On coreweave-rtx6000, a 0-GPU job takes a GPU node's spare share, which on a node whose GPUs are all held is 32 CPUs and 242 GB (flyloft's `docs/design/spare-cpus.md`, checked 2026-11-04). So the skill's example asks `--partition=hpc-low --cpus-per-task=32 --mem=200G`. A CPU job waits for a node with that much spare; the skill says so. Search time for a batch is unknown until the first real batch; the skill records it then.

**Running it** (on coreweave-rtx6000, from the directory holding the FASTA):

```bash
srun --partition=hpc-low --cpus-per-task=32 --mem=200G --time=12:00:00 \
  --container-image=ghcr.io#brineylab/toolshed/colabfold-search:1.0.0 \
  --container-mounts="$PWD":/work,/mnt/data/datasets/colabfold-db/v1.0.0:/db:ro --container-workdir=/work \
  colabfold_search --threads 32 --filter 0 seqs.fasta /db msas
```

**Gotchas the skill states.**
- The two command lines give different MSAs by design; an evaluation uses one of them throughout and names it.
- CPU and (future) GPU search give different MSAs; never mix them in one evaluation.
- `--filter 0` MSAs can be very large for well-represented families; the skill gives measured sizes after the first batch.
- `/db` must be a `READY` copy (§5); a half-unpacked copy gives wrong or missing hits without an error.
- **MMseqs2 #1092.** Release 18 dies in the k-mer prefilter on conserved k-mers at scale (it killed `easy-cluster` on about 1.1M antibody pairs; `datascience` builds MMseqs2 from master commit `2a3dafc` for the fix). Whether `colabfold_search`'s prefilter, with antibody queries against a database holding many antibodies, meets it is unknown. Release 18 stays the default because it matches the server; if a real batch dies with that signature, the next version builds MMseqs2 from the fixed commit and the skill says its MSAs may then differ slightly from the server's.

## 5. Putting a version on a cluster

Written as prose and full command lines in `colabfold-search`'s skill **(call)**: nothing new is built in either repository, and it runs once per cluster per `colabfold-db` version.

1. **Pull** into the cluster's shared Hub cache, `<data_dir>/hf_hub`, the one flyloft's jobs read (flyloft `docs/design/datasets.md` §4; the datasets README's `hub-cache` disagrees and is fixed in the datasets pull request **(call)**), in `datascience` (the only step with network):
   `srun --partition=hpc-low --cpus-per-task=4 --mem=8G --container-image=ghcr.io#brineylab/toolshed/datascience:1.1.0 --container-mounts=/mnt/data/datasets:/mnt/data/datasets --export=ALL,HF_HUB_CACHE=/mnt/data/datasets/hf_hub python3 -I -c 'from huggingface_hub import snapshot_download; print(snapshot_download("brineylab/colabfold-db", repo_type="dataset", revision="v1.0.0"))'`
2. **Unpack** inside `colabfold-search`, with no network, into `/mnt/data/datasets/colabfold-db/v1.0.0.tmp/`; then rename it to `/mnt/data/datasets/colabfold-db/v1.0.0/` and write `READY` inside it last. A copy without `READY` is incomplete and is deleted and redone. The rename makes a finished copy appear all at once; two people unpacking at once is not guarded against **(call)**: the skill says to check for the directory first.
3. **Search** jobs mount `/mnt/data/datasets/colabfold-db/v1.0.0` read-only at `/db`.

No index is built: ColabFold's README says CPU batch search does not need one and may be slowed by one. The unpacked size is measured on the first unpack and written into both skills. The unpacked copy is derived from the Hub version and can be rebuilt from it, like the Hub cache it sits beside. The tarballs in the Hub cache may be deleted after unpacking to save 236 GB; the skill says so and says that a re-unpack then pulls again.

The builder confirms with the cluster's admins that `/mnt/data/datasets/` is writable by the person doing this and has room for the cache plus the unpacked copy.

## 6. Proposed amendments

**Toolshed `docs/DESIGN.md`.**
- §1, the "Weights are in the image" bullet, after "The ceiling is databases too big for an image (AlphaFold's); those belong to the data companion, not the shed." add: "A tool that searches such a database (`colabfold-search`) holds its programs and no database: its skill names the dataset and version it is documented against, says how a version is unpacked once per cluster onto shared storage, and its jobs mount that copy read-only. A result is named by the tool's version and the dataset's. Its test builds a tiny database from its fixture and still passes with no network."
- §7, the row "Bulk databases (AlphaFold's) | The data companion's design": picked up 2026-11-09 by `colabfold-search` and `colabfold-db`; remove the row.
- §7, new row: "A tool that serves requests on a port (an MSA server that structure predictions call while they run) | When a job learns its sequences only while it runs, so its MSAs cannot be searched in a batch first".
- The status line gains the amendment.

**Datasets `docs/DESIGN.md`.**
- §1, after "The files are what training and evaluation read directly, parquet where the data is tabular, sharded when a file would be large": add "A database a tool searches (`colabfold-db`) is the exception: its files are the upstream archives byte for byte, the tool's skill says how a version is unpacked once per cluster beside the shared Hub cache, and the unpacked copy, like the cache, is derived and disposable. Its test checks the archives' sizes and headers in seconds; the recipe checked their hashes and the record carries them."
- The status line gains the amendment.

The datasets repository's `AGENTS.md` says files are "parquet (sharded when large)"; its rule list gains the same exception in one clause.

## 7. Deferred

| Item | Picked up when |
|---|---|
| An MSA server on a port | A job learns its sequences only while it runs (a design loop, a campaign generating sequences) |
| GPU search (MMseqs2-GPU): a GPU section, the NVIDIA variables, the GPU binary, an index built once per cluster under `/mnt/data/datasets/colabfold-db/<version>/gpu-<tool version>/` with its own `READY`, and a GPU test | A CPU batch is too slow for the work that needs it |
| Mirroring a built GPU index on the Hub | A third cluster needs it, or building it takes longer than downloading it would |
| Paired MSAs for complexes (`uniref30_2302_newtaxonomy` in a new `colabfold-db` version, `:`-joined input, Boltz-2's paired CSV) | A complex whose chains' pairing carries signal is predicted; antibody heavy/light pairing by taxonomy carries little |
| Templates (PDB100) | A structure tool that uses templates needs them |
| A copy on `bcr` | Work there needs it; the same §5 steps under its own paths |
| Raised sensitivity (`-s`) for the coevolution line | The first batch's Neff shows depth limits P@L |
| A lock around unpacking | Two people are seen unpacking the same version at once |

## 8. Order of work

1. The datasets pull request (`colabfold-db`), including any change `scripts/push` needs for 236 GB, and the datasets amendment. It is the long pole: download and upload.
2. The toolshed pull request (`colabfold-search`), with its record from a test on coreweave-rtx6000, and the toolshed amendment. Its test does not need the dataset, so it can be built in parallel; its skill names the dataset version from step 1.
3. §5's steps on coreweave-rtx6000, then the first real batch of each kind, whose measured time, memory, disk, MSA sizes, and (for coevolution) Neff go into the skill as a version bump.

## 9. Left to the builder

- How `colabfold_search` names its outputs, and the tiny-database recipe for the test.
- Whether `colabfold` runs without its JAX extras, and how the entry point is isolated.
- Exact pins (base digest, `colabfold` version, MMseqs2 binary sha256).
- Whether `scripts/push` handles 236 GB, and the fix if not.
- Confirming the databases' license and the shared storage's write access and room.
