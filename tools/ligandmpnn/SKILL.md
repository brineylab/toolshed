---
name: ligandmpnn
description: "LigandMPNN inverse folding: designs protein sequences onto a fixed backbone while conditioning on neighboring non-protein atoms, such as redesigning an antibody around a bound antigen, a pocket around a ligand, or a protein around DNA or metals. Reach for it when sequence design on an existing structure is needed; it returns designed sequences with per-design confidence and sequence-recovery numbers, not folded structures."
metadata:
  version: 1.1.0
---
# ligandmpnn

Image: `ghcr.io/brineylab/toolshed/ligandmpnn:1.1.0`

LigandMPNN (the `dauparas/LigandMPNN` repository, MIT licensed, at commit `26ec57ac976ade5379920dbd43c7f97a91cf82de`; its checkpoints are published by the Institute for Protein Design without a license of their own, and the repository's MIT license is what publishes them — the upstream `LICENSE` file ships in the image at `/opt/LigandMPNN/LICENSE`), run through upstream's own `run.py` on the shed's CUDA-13 PyTorch 2.14.1 image (Python 3.12) with numpy 2.5.4 and ProDy 2.6.1 — no ProDy release before 2.6.1 ships a Linux wheel, and its parse/select/write API is unchanged from the 2.4.1 upstream pinned — plus dm-tree and ml-collections, which the code's vendored openfold imports. The wrapper `ligandmpnn` runs `run.py` from any directory and fills in the image's checkpoint paths and a `--model_type ligand_mpnn` default, so the command below is the whole tool.

## Commands

A design run, sampling 32 sequences on one PDB:

```bash
ligandmpnn --model_type ligand_mpnn --pdb_path complex.pdb --out_folder out --temperature 0.2 --seed 1 --batch_size 16 --number_of_batches 2
```

Fixing residues (kept exactly as they are; format is space-separated chain+residue-number) while redesigning the rest of chain C, with the side-chain packer and per-design stats on:

```bash
ligandmpnn --model_type ligand_mpnn --pdb_path complex.pdb --out_folder out --fixed_residues "C1 C2 C3 C4 C5" --chains_to_design C --temperature 0.2 --seed 1 --batch_size 8 --number_of_batches 1 --save_stats 1 --pack_side_chains 1
```

A batch over many PDBs, where `pdbs.json` is a JSON object whose keys are paths (values are ignored, `""` is fine):

```bash
ligandmpnn --model_type ligand_mpnn --pdb_path_multi pdbs.json --out_folder out --temperature 0.2 --seed 1 --batch_size 4 --number_of_batches 1
```

Other options that matter: `--redesigned_residues "C45 C46"` redesigns only those (the inverse of `--fixed_residues`); `--chains_to_design "H,L"` redesigns only those chains, all other protein chains fixed (comma-separated); `--parse_these_chains_only "A,B"` parses only those chains from the PDB, dropping the rest from context; `--omit_AA "CM"` and `--bias_AA "C:-1.0,P:2.0"` bias the amino acids sampled; `--pack_side_chains 1` also runs the shipped side-chain packer (`--number_of_packs_per_design 4`, `--repack_everything 0`). `run.py`'s `--help` lists every option.

## Inputs and outputs

- **In:** PDB files — one via `--pdb_path`, or many via `--pdb_path_multi` and a JSON object of paths. Every non-protein atom in each parsed chain set (small molecules, DNA, RNA, metals) becomes the ligand context the model conditions on; waters are dropped. Insertion codes are honored in `--fixed_residues`-style tokens (chain, number, icode).
- **Out:** under `--out_folder`:
  - `seqs/<pdb-name>.fa` — first record is the native sequence, its header carrying `num_res` (designed residues) and `num_ligand_res` (size of the parsed ligand context); then one record per design: header `><name>, id=<k>, T=<temp>, seed=<seed>, overall_confidence=<x>, ligand_confidence=<x>, seq_rec=<x>`, then the sequence, chains joined by `:` (`--fasta_seq_separation`). `overall_confidence` is the design's self-consistency under the model, `ligand_confidence` the same over residues with context atoms within 8 Å (`--ligand_mpnn_cutoff_for_score`), and `seq_rec` the sequence recovery against native. All are on (0, 1], higher is better.
  - `backbones/<pdb-name>_<id>.pdb` — the input backbone with the designed sequence written as residue names (no side chains).
  - with `--save_stats 1`: `stats/<pdb-name>.pt` — a dict of `generated_sequences`, `sampling_probs`, `log_probs`, `decoding_order` tensors.
  - with `--pack_side_chains 1`: `packed/<pdb-name>_packed_<id>_<k>.pdb` — full-atom structures with packed side chains, `k` running over `--number_of_packs_per_design` samples per design.

## Compute

One GPU, and a light one: the checkpoints are 7–14 MB and designs take seconds. The fixture of this tool's test (1BC8, 93 residues with two DNA chains and two zinc ions as context) yields 44 designs plus 32 packed structures within minutes on one L40S; antibody-scale Fabs run the same way, batch size the limit. CPU-only runs work but are slower.

## Running it

On a laptop, from the directory holding `complex.pdb`:

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/ligandmpnn:1.1.0 \
  ligandmpnn --model_type ligand_mpnn --pdb_path complex.pdb --out_folder out --temperature 0.2 --seed 1 --batch_size 16 --number_of_batches 2
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/ligandmpnn:1.1.0 \
  --container-mounts="$PWD":/work --container-workdir=/work \
  ligandmpnn --model_type ligand_mpnn --pdb_path complex.pdb --out_folder out --temperature 0.2 --seed 1 --batch_size 16 --number_of_batches 2
```

## Gotchas

- Upstream pins numpy 1.23.5 because this commit's vendored openfold still names the `np.int` dtype alias, which numpy 1.24 removed, and Python 3.12 has no numpy that old. The image installs numpy 2.5.4 and patches the three module-level dtype declarations in `openfold/np/residue_constants.py` to the built-in (`dtype=int` — the same array the alias produced); everything else is upstream's code as committed, and the build asserts the patch still applies.
- Upstream's `--model_type` default is `protein_mpnn`, which designs without any ligand context. The wrapper inserts `--model_type ligand_mpnn` when none is given; pass `--model_type protein_mpnn` explicitly for context-free design, and `--checkpoint_ligand_mpnn`/`--checkpoint_protein_mpnn`/`--checkpoint_path_sc` to pick other checkpoints. The image also carries the other fixed-residue-mix checkpoints (`ligandmpnn_v_32_005_25`, `_020_25`, `_030_25` — trained with 5/20/30% of residues fixed, versus the shipped default `_010_25`'s 10%).
- `--seed 0` is upstream's sentinel for "no seed": the run then picks a random seed (recorded in the output header), so designs are irreproducible. Pass a nonzero seed when reproducibility matters.
- LigandMPNN's checkpoints are plain `torch.save` dictionaries, which torch ≥ 2.6 refuses under its `weights_only=True` default, so the image sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. It is inherited by anything else run in the container.
- Residue tokens (`--fixed_residues`, `--redesigned_residues`) are author numbering: `C1`, `C2`, matching the PDB's own residue numbers and insertion codes, not zero-based indices (`--zero_indexed 1` only renumbers outputs). Chain lists (`--chains_to_design`, `--parse_these_chains_only`) are comma-separated; residue lists are space-separated.
- Waters never reach the ligand context; everything else non-protein does, including metals and whole DNA chains — for an antibody–antigen design, pass `--chains_to_design "H,L"` on the complex PDB so the antigen is context and only the antibody is redesigned.
- The wrapper runs `run.py` under `python3 -E -s`: not `-I`, which would drop `run.py`'s own directory from `sys.path` and break its `data_utils` imports. A caller's `PYTHONSTARTUP`, `PYTHONPATH`, and user site-packages are still ignored.
- The fixture crosses no runtime-compilation threshold (LigandMPNN has none), but it does exercise every documented path with real context: the ligand-atom encoders over 20 DNA nucleotides and 2 zinc ions, `--fixed_residues` with `--chains_to_design`, `--pdb_path_multi`, `--save_stats`, and the side-chain packer with its own checkpoint. Multi-model PDBs (`MODEL` records) are not exercised by the test.
