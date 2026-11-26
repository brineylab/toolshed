---
name: esm-if1
description: "ESM-IF1 structure-conditioned inverse folding: designs and scores protein sequences from N/CA/C backbones, with selected-chain or all-protein-chain context, and extracts 512-dimensional per-residue GVP embeddings. Reach for it when a sequence must be judged against a crystal or predicted structure, including antibody chains with their antigen backbone in view; it does not predict structures."
metadata:
  version: 1.0.0
---
# esm-if1

Image: `ghcr.io/brineylab/toolshed/esm-if1:1.0.0`

ESM-IF1 (`esm_if1_gvp4_t16_142M_UR50`, 142M parameters), from `facebookresearch/esm` at commit `2b369911bb5b4b0dda914521b9475cad1656b2ac`, on PyTorch 2.14.1 CUDA 13, PyG 2.8.1, and biotite 0.41.2. This is the GVP-Transformer trained on CATH and 12M AlphaFold2-predicted structures, not an ESM-2 sequence-only model. The weights, upstream source, and license are in the image; a run needs no network.

## Commands

A single-chain design run, two independently seeded samples, and a structure-only embedding run:

```bash
esm-if1 design --structure single.pdb --chain C --out designs.fa --num-designs 2 --temperature 0.1 --seed 1
esm-if1 score --structure single.pdb --chain C --sequences candidates.fa --out scores.tsv --per-residue residue-scores.npz
esm-if1 embed --structure single.pdb --chain C --id single --out embeddings.npz
```

Condition a heavy chain on just the light chain and antigen, score the same sequence with the context, or use every protein chain in the file:

```bash
esm-if1 design --structure antibody.cif --chain H --context-chains L P --out designs.fa --id antibody --num-designs 1 --temperature 0.1 --seed 7
esm-if1 score --structure antibody.pdb --chain H --context-chains L,P --sequences candidates.fa --out scores.tsv
esm-if1 score --structure antibody.cif --chain H --multichain-backbone --sequences candidates.fa --out scores.tsv
esm-if1 embed --structure antibody.cif --chain H --multichain-backbone --id antibody --out embeddings.npz
```

All commands accept `--device cuda` (the default when a GPU is visible) or `--device cpu`; one structure and chain run at a time. Design takes `--num-designs` (default 1), `--seed` (default 1), and `--temperature` (default 0.1, suitable for recovery-oriented designs; around 1 makes more diverse, less native-like sequences). `--id` replaces the structure filename stem in the output.

## Inputs and outputs

- **In:** first-model PDB or `.cif`/`.mmcif`. Chains are **author** (`auth_asym_id`) ids, mmCIF included, not label asym ids. `--chain` can be omitted only if there is one protein chain. Context is protein N/CA/C backbones only; nucleic acids, ligands, metals and waters are ignored. Use `--context-chains` for named chains (whitespace- or comma-separated) or `--multichain-backbone` for every protein chain, not both. The target comes first in the model input and cannot repeat in the context; ten NaN-separated residues divide it from context. Nothing *not represented by an atom record* is invented, but a residue with a side-chain atom and no N/CA/C is retained with those coordinates unresolved. If one of N/CA/C is missing, all three are marked unresolved for that residue.
- **`design`:** FASTA follows LigandMPNN's output shape: first the native sequence, header `name chain=C num_res=L native=1`; then each design, `name, id=k, T=t, seed=s, overall_confidence=x, seq_rec=x`. `overall_confidence` is `exp(ll_fullseq)`, the geometric mean target residue probability, in (0,1]; `seq_rec` is the fraction of residues matching the deposited native at that position. `ll_fullseq` is the average autoregressive log-likelihood computed under the exact same context as sampling. Sequences are 20 canonical amino acids, not multiple chains.
- **`score`:** FASTA ids (first whitespace-delimited header token) must be unique, and each sequence must match the target coordinate count exactly and use the 20 canonical residues. Output TSV is `id`, `ll_fullseq`, `ll_withcoord`: natural-log means over the whole target and over target residues with complete finite N/CA/C. Higher (less negative) is more likely. These are per-residue averages, not length-summed log-likelihoods or per-position mutation effects. `--per-residue` `.npz` gives `ll_<id>` float32 arrays, `ids`, and the `resolved_coordinates` Boolean vector for the shared target. The TSV and NPZ written by the same invocation agree.
- **`embed`:** one `.npz` with `ids` (length-1 string array), `chains` (the one target chain), and `embeddings`, float32 `(L,512)`: the final structure-encoder output for the target positions, with selected context seen by that encoder. It is not a sequence model embedding or mean-pooled.

## Compute

One GPU, ~1.7 GB of checkpoint parameters on disk plus inference buffers; an A100 is the intended hardware. There is no crop: cost grows with the **target plus selected context**, not target length alone, and sampling also decodes placeholders over context as upstream does. For large assemblies select only relevant chains instead of `--multichain-backbone`; checkpoint loading is several seconds, so do not start a fresh CLI process on every residue. Short complexes can run on CPU, much more slowly. Multimodel inputs read only model 1.

## Running it

```bash
podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work ghcr.io/brineylab/toolshed/esm-if1:1.0.0 esm-if1 design --structure antibody.pdb --chain H --context-chains L P --out designs.fa --num-designs 2 --temperature 0.1 --seed 1
```

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/esm-if1:1.0.0 --container-mounts="$PWD":/work --container-workdir=/work esm-if1 score --structure antibody.cif --chain H --multichain-backbone --sequences candidates.fa --out scores.tsv
```

For offline operation add `--network none` (podman); no Hugging Face cache or token is used.

## Gotchas

- `--context-chains` affects the **structure encoder**; only the target sequence is designed or scored, and non-target decoder tokens are `<pad>`. Upstream's `score_sequence_in_complex` omits the pads needed to match concatenated coordinate length; the wrapper adds them. Context does not mean its sequence tokens are given to the decoder. `overall_confidence` is a mean-probability transform, not raw LL: the TSV holds raw log-likelihoods.
- Select author ids exactly as the structure has them. A duplicated `H1/H2` label segmentation in mmCIF is naturally reunited under `H` with `use_author_fields=True`; arbitrary or missing chains are refused rather than guessed. `--multichain-backbone` means every parsed protein chain (it might include symmetry copies or extra partners), not only the other named antigens. Unsupported nonstandard amino acids require supplying the native sequence from the same parser rather than treating `X` as a canonical input; scores reject `X` explicitly. Fully absent residues still require a caller-provided sequence/structure alignment and are not reconstructed.
- Original temperature 1 samples may be diverse at the expense of recovery; use 0.1 or lower for the native-recovery check in the validation campaign. Reproducibility of the seed includes the model's nondeterministic CUDA kernel caveat: the test compares numerical scores/embeddings to its reference at tolerance, not a string-identical sample.
- The image has PyTorch's legacy `torch.load` policy enabled because upstream's public checkpoint is a torch.save dictionary, as with the shed's older inverse folder. The optional PyG `torch_scatter` path is not used by the ESM-IF1 GVP graph layer; the package is pure Python on this path.
- Test scope: compact 93-residue 1BC8 C crops (smoke), plus the actual 520-residue 2JEL L/H/P antibody-antigen complex in both file formats (520 target/context positions plus separators), so native and shuffled sequences, selected/all context, unresolved H12, and 512-D encoder output all run on a realistic complex. It assumes no residues beyond ordinary antibody lengths, no enormous assembly sampling, no amino-acid alphabet outside the canonical 20, and no runtime package compilation. Target sequences such as the original un-deposited tails are not added speculatively.
- Licensing: upstream code is MIT. The checkpoint has no separate license file; following the toolshed rule, the repository's MIT license accompanies it at `/opt/model/LICENSE`. 1BC8 and 2JEL fixtures are PDB archive data.
