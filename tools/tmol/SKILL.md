---
name: tmol
description: "tmol, RosettaCommons' GPU implementation of the Rosetta energy function in PyTorch: score, Cartesian-minimize, repack, and FastRelax protein and ligand structures, one file or a whole directory batched on one GPU. Reach for it to rescore or refine structures with the beta2016 energy; anything past the four verbs' defaults — per-term scores, movement masks, restraints, gradients — is a short python3 -I script against the same library, documented below."
metadata:
  version: 1.0.0
---
# tmol

Image: `ghcr.io/brineylab/toolshed/tmol:1.0.0`

tmol 1.0.0 — the RosettaCommons `tmol` package, BSD-3-Clause (Institute for Protein Design; the upstream `LICENSE` ships in the image at `/opt/tmol/LICENSE`; documentation, quickstart, and runnable tutorials at https://github.com/RosettaCommons/tmol/tree/v1.0.0) — on the shed's CUDA-12.6 PyTorch 2.14.1 image (Python 3.12), with the `cu12` extra (`tmol-ops-cu12`, the CUDA operators) and the `ligand` extra (Open Babel, which ligand preparation uses). The beta2016 parameters ship inside the wheel: no weights are fetched, the image needs no network, and it runs on CPU too (`--device cpu`), slower there (measured below). tmol's scores are tmol score units — not kcal/mol, and not numerically interchangeable with Rosetta's. Upstream ships no command-line program, so the image carries the shed's wrapper, `tmol`, four verbs with defaults; everything past the defaults is the library way, `python3 -I script.py`, below.

## Commands

The four verbs. `<in>` is one structure file or a directory of them (PDB, mmCIF, MOL2, or SDF, optionally `.gz`/`.zst` compressed); a directory's files load into one batch, and everything is processed together on the device.

```bash
tmol score in/ scores.tsv
tmol minimize in/ out/
tmol pack in/ out/ --seed 1
tmol relax in/ out/
```

- `score` writes a TSV, one row per input — `name`, `total`, then one column per weighted beta2016 term, `%.6f` values. Every verb scores with beta2016.
- `minimize` is Cartesian minimization with every atom free; `pack` is fixed-sequence side-chain repacking; `relax` is FastRelax with its defaults (Cartesian, every atom free).
- The optimizer verbs write one structure per input into `out/` — the input's own format and file name, compression stripped — plus `out/scores.tsv`, the same numbers `tmol score out/` reports again to within GPU reduction noise (measured ≤ 2×10⁻⁴ on the total; not byte-identical), so the verbs chain through files:

```bash
tmol relax in/ out/
tmol score out/ scores.tsv
```

- Every verb takes `--device DEV` (default: cuda when one is visible, else cpu); `pack` and `relax` also take `--seed N` (default 0). That is the whole surface: no other flags, no other score function, and the wrapper sequences nothing.

The second way in is the library itself, under `python3 -I` — the same calls the wrapper makes. A script that loads a structure, relaxes it, and scores it again, all in one process (`pose.pdb` in, `relaxed.pdb` out):

```bash
python3 -I relax_score.py pose.pdb relaxed.pdb
```

`relax_score.py`:

```python
import sys

import torch
from tmol import (
    beta2016_score_function,
    fast_relax,
    file_from_pose_stack,
    pose_stack_from_file,
)

source, output = sys.argv[1], sys.argv[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

pose = pose_stack_from_file(source, device, packer_seed=0)
sfxn = beta2016_score_function(device)
scorer = sfxn.render_whole_pose_scoring_module(pose)
with torch.no_grad():
    before = scorer(pose.coords).tolist()

relaxed = fast_relax(pose, sfxn)

# Re-render the scorer on the relaxed pose: its atom layout can change,
# and the old scorer is bound to the old layout.
scorer = sfxn.render_whole_pose_scoring_module(relaxed)
with torch.no_grad():
    after = scorer(relaxed.coords).tolist()

file_from_pose_stack(relaxed, output)
print(f"before {before[0]:.6f}")
print(f"after {after[0]:.6f}")
```

Past the verbs' defaults, the pieces worth knowing:

**Loading and batching.** `pose_stack_from_file(path, device, packer_seed=0)` loads one file into a one-pose stack. Several files that belong together — a directory, a protein and its ligands — load through one shared context, which is also how the wrapper loads a directory:

```python
import biotite.structure
from tmol import PoseStackBuilder, atom_array_from_file, pose_stack_from_atom_array
from tmol.io import build_context_from_atom_array

arrays = [atom_array_from_file(str(path)) for path in paths]
context = build_context_from_atom_array(
    biotite.structure.concatenate(arrays), device,
    prepare_ligands=True, ligand_seed=0,
)
poses = [
    pose_stack_from_atom_array(array, device, context=context, packer_seed=0)
    for array in arrays
]
stack = PoseStackBuilder.from_poses(poses, device)
```

`prepare_ligands=True` parameterizes whatever noncanonical chemistry the inputs hold — small molecules, metals, nucleic acids — and is how protein–ligand complexes score. The score function must then be built from the context's database: `beta2016_score_function(device, param_db=context.parameter_database)`. Protein-only inputs can instead use plain `pose_stack_from_file` with `beta2016_score_function(device)`, which share tmol's default database. Waters (HOH) are dropped on load, in every path.

**Per-term scores.**

```python
scorer = sfxn.render_whole_pose_scoring_module(stack)
with torch.no_grad():
    totals = scorer(stack.coords)                    # [n_poses]
    terms = scorer(stack.coords, sum_terms=False)    # [n_terms, n_poses]
names = [score_type.name for score_type in sfxn.all_score_types()]
```

`sum_terms=False` returns weighted terms; add `apply_weights=False` for the unweighted decomposition. `render_block_pair_scoring_module()` attributes the score block by block. Render a scorer again after anything that can change the stack's atom layout (packing, relaxation); a coordinate-only update with the same layout reuses it.

**Minimization.** `run_cart_min(stack, sfxn)` optimizes coordinates with every atom free (the `minimize` verb). Pass a boolean mask with `stack.coords`' shape to move only chosen atoms. Torsions and rigid-body jumps are `run_kin_min`'s territory, with a configured `FoldForest` and `MoveMap` — a `MoveMap` enables nothing by existing; set its flags.

**Packing.** Fixed-sequence repacking, the `pack` verb's recipe:

```python
from tmol import PackerPalette, PackerTask, pack_rotamers
from tmol.pack import (
    FixedAAChiSampler,
    IncludeCurrentSampler,
    create_dunbrack_sampler_from_database,
)

task = PackerTask(stack, PackerPalette())
task.restrict_to_repacking()
task.add_conformer_sampler(
    create_dunbrack_sampler_from_database(context.parameter_database, device)
)
task.add_conformer_sampler(FixedAAChiSampler())
task.add_conformer_sampler(IncludeCurrentSampler())
packed = pack_rotamers(stack, sfxn, task, seed=1)
```

`restrict_to_repacking()` keeps the sequence; design needs an explicit allowed-identity task. `disable_packing_by_block_mask()` freezes chosen blocks. `pack_rotamers` chunks large stacks itself and its `seed=` forks torch's generator; render a new scorer on the packed stack.

**FastRelax.** `relax` is `fast_relax(stack, sfxn)` — Cartesian, every atom free. The `minimizer=` argument carries the mode and its settings:

```python
from tmol import KinematicRelaxMinimizer, fast_relax

torsions = fast_relax(stack, sfxn, minimizer=KinematicRelaxMinimizer())
```

`CartesianRelaxMinimizer(cartesian_move_map)` and `KinematicRelaxMinimizer(move_map, fold_forest)` each reject the other's move-map type; without a move map the kinematic minimizer frees every torsion and jump, while `MoveMap.from_pose_stack` frees nothing until enabled. `cuda_graph=False` forces eager execution. `fast_relax` takes no seed argument — `torch.manual_seed(N)` right before it, which is what the wrapper does.

**Constraints.** A constraint must be attached to the pose and its score term must carry a nonzero weight in the score function; miss either and it is silently inactive.

**Coordinate gradients.** The score is differentiable in the coordinates:

```python
coords = stack.coords.detach().clone().requires_grad_(True)
total = scorer(coords).sum()
total.backward()
gradient = coords.grad    # stack.coords' shape
```

## Inputs and outputs

- **In:** one structure file, or a directory of them (read flat, not recursively): PDB (`.pdb`, `.ent`), mmCIF (`.cif`, `.mmcif`, `.bcif`), MOL2, SDF, or MDL `.mol`, each optionally `.gz`, `.gzip`, or `.zst`. Proteins, nucleic acids, ligands, and complexes; waters are dropped.
- **Out:** `score` writes its TSV with a `name` column (the input's file name), `total`, and one column per weighted beta2016 term. The optimizer verbs write one structure per input — same format, same file name, compression stripped — plus `scores.tsv` in the same shape, which `tmol score <outdir>` reproduces to within GPU reduction noise (≤ 2×10⁻⁴ measured on the total).

## Compute

One GPU. Measured end to end — load, compute, write, and the written structures' rescore — on the test's fixture (26 poses of 1UBQ, 72 KB a pose) on one NVIDIA L40S (46 GB, driver 580.178.04), torch 2.14.1+cu126:

| verb | time | rate |
| --- | --- | --- |
| `score` | 23–24 s | ~1.1 structures/s |
| `minimize` | 28 s | ~0.9 structures/s |
| `pack` | 16–18 s | ~1.5 structures/s |
| `relax` | 38–39 s | ~0.7 structures/s |

The library way's one-pose relax-then-score script took 14 s. Most of a small run is fixed cost — structure load, parameter-database rendering, first-use CUDA compilation — so rates rise with batch size until the packer's chunking (25 poses a chunk here) and GPU memory bind; every verb prints its own load / operate / write / score breakdown to stderr. On CPU (16 cluster cores) the same batch ran `score` 29 s, `minimize` 29 s, `pack` 21 s, `relax` 72 s — 1.0–1.9× slower end to end, the gap widening with the compute's share — and the one-pose script 77 s (5.6× slower: a single pose's tensors leave the CPU cores idle where a batch spreads over them). The fixture's CPU and GPU totals agree within the same ~3×10⁻⁴ reduction noise.

## Running it

On a laptop, from the directory holding `in/` (the wrapper picks the GPU when one is visible, else CPU):

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/tmol:1.0.0 tmol score in/ scores.tsv
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/tmol:1.0.0 \
  --container-mounts="$PWD":/work --container-workdir=/work \
  tmol relax in/ out/
```

## Gotchas

- The test's fixture is 26 copies of water-stripped 1UBQ because tmol's packer chunks a pose stack: a pose of at most 256 blocks (1UBQ's 76) packs in chunks of 25 poses, so 26 copies crosses the boundary — the test exercises the chunked path (25 poses, then 1); fewer than 25 poses never leaves one chunk. `TMOL_PACK_MAX_POSES_PER_CHUNK` lowers the bound when GPU memory is tight.
- tmol's scores are tmol score units, not kcal/mol, and not numerically interchangeable with Rosetta's; compare tmol numbers with tmol numbers.
- Waters (HOH) are silently dropped on load, in every path; everything else non-protein — ligands, metals, DNA — is parameterized and kept.
- Seeding: every verb seeds torch and the load-time hydrogen packer; `--seed` (pack/relax, default 0) additionally drives the packer's annealer and FastRelax. `score` always runs at seed 0, and the optimizer verbs score their written outputs at seed 0, which is why `tmol score <outdir>` reproduces `scores.tsv` within the reduction noise above. `fast_relax` takes no seed argument — library callers `torch.manual_seed` before it.
- Seeded reproducibility, measured on the fixture: two `tmol pack --seed 1` runs wrote byte-identical structures (their `scores.tsv` totals 1.6×10⁻⁴ apart — the reduction noise), but two `tmol relax --seed 1` runs did **not**: CUDA score reductions are non-deterministic at the ~10⁻⁴ level, and FastRelax's iterated pack-and-minimize cycles amplify that noise into different local minima — the pair's totals differed by up to 3.1 and a few side-chain atoms by up to 5.4 Å, pose for pose, at the same seed. Treat `relax` as stochastic: compare energies, not coordinates, and take the best of several seeds when the structure matters.
- The image is CUDA 12.6: it needs an NVIDIA driver from the CUDA 12 era (≥ 525.60.13). The campaign measured on an L40S with driver 580.178.04.
- On CPU (`--device cpu`) every verb works and the totals agree with the GPU's within the same ~3×10⁻⁴ noise, but the compute falls behind as the batch shrinks: 1.0–1.9× slower per verb on the 26-pose fixture and 5.6× on a single pose (see Compute).
- The wrapper runs under `python3 -I`, and the library way above is the same flag: a caller's `PYTHONPATH` or user site-packages never replace what the image tested.
- A directory is read flat (no recursion) and only structure-suffixed files are collected, so the `scores.tsv` inside an optimizer verb's outdir does not confuse a later `tmol score` of that directory.
