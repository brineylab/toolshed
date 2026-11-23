"""Score an ablm-forge checkpoint with AbLM-Eval, and check the answer.

Forge checkpoints carry no auto_map: they resolve only because importing
ablm-forge registers its "ablm" architecture, which is why the image installs
ablm-eval[forge].  No forge checkpoint is public, so the test writes one: a
tiny AblmForMaskedLM, seeded, saved with save_pretrained and no tokenizer files,
as forge's pretrain script and Trainer write their snapshots.  AbLM-Eval's
Python API (evaluate_ablms, the documented way to score a checkpoint on disk;
the CLI takes Hub repositories at a pinned revision) runs per_position on it
over the paired fixture, through its loader and its strict weight check.  The
known answer is computed here, independently, from the model in memory before
it was saved: each residue masked in turn, the true residue's log-probability.
The two must agree to 1e-4.

Usage: forge.py FIXTURE WORKDIR
"""

import json
import sys
from importlib.metadata import distribution, version
from pathlib import Path

import pandas as pd
import torch

import ablm  # noqa: F401  (registers the "ablm" architecture)
from ablm import AblmConfig, AblmForMaskedLM, AblmTokenizerFast
from ablm_eval import PerPositionConfig, evaluate_ablms
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

TOLERANCE = 1e-4
SEPARATOR = "<cls><cls>"


def reference(model, tokenizer, sequences, device):
    special = set(tokenizer.all_special_ids) - {tokenizer.unk_token_id}
    first_ids, per_sequence, losses = None, [], []
    for sequence in sequences:
        ids = torch.tensor(tokenizer(sequence)["input_ids"])
        first_ids = first_ids or ids.tolist()
        positions = [i for i, t in enumerate(ids.tolist()) if t not in special]
        masked = ids.repeat(len(positions), 1)
        masked[range(len(positions)), positions] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(input_ids=masked.to(device)).logits.float()
        log_probs = logits[range(len(positions)), positions].log_softmax(-1)
        row = log_probs[range(len(positions)), ids[positions].to(device)].tolist()
        per_sequence.append(sum(row) / len(row))
        losses.extend(row)
    return first_ids, per_sequence, sum(losses) / len(losses)


def main() -> None:
    fixture, work = sys.argv[1], Path(sys.argv[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forge = json.loads(distribution("ablm-forge").read_text("direct_url.json"))
    print("ablm-forge", version("ablm-forge"), "at", forge["vcs_info"]["commit_id"])
    assert "ablm" in CONFIG_MAPPING, "importing ablm-forge did not register its architecture"

    torch.manual_seed(0)
    config = AblmConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=512
    )
    model = AblmForMaskedLM(config).eval()
    checkpoint = work / "forge-tiny"
    model.save_pretrained(checkpoint)
    print("checkpoint files:", sorted(p.name for p in checkpoint.iterdir()))

    evaluate_ablms(
        {"forge-tiny": str(checkpoint)},
        [
            PerPositionConfig(
                data_path=fixture,
                antibody_datatype="paired",
                dataset_name="mini",
                separator=SEPARATOR,
                num_proc=1,
            )
        ],
        shared_output_dir=str(work / "results"),
        generate_comparisons=False,
    )
    (result_file,) = (work / "results").rglob("*per-position-inference.parquet")
    df = pd.read_parquet(result_file)

    data = pd.read_parquet(fixture)
    sequences = [h + SEPARATOR + l for h, l in zip(data.sequence_aa_heavy, data.sequence_aa_light)]
    first_ids, per_sequence, mean_log_prob = reference(
        model.to(device), AblmTokenizerFast(), sequences, device
    )

    got_ids = [int(t) for t in df["tokenized_sequence"].iloc[0]]
    print("token ids of the first paired input:", got_ids)
    assert got_ids == first_ids, f"AbLM-Eval tokenized {got_ids}, forge's tokenizer {first_ids}"
    assert len(df) == len(sequences), f"{len(df)} sequences scored, not {len(sequences)}"
    got_per_sequence = [-sum(row) / len(row) for row in df["loss"]]
    losses = [x for row in df["loss"] for x in row]
    got_mean = -sum(losses) / len(losses)
    for got, want in zip(got_per_sequence, per_sequence):
        assert abs(got - want) < TOLERANCE, f"a sequence's mean {got:.6f} is not {want:.6f}"
    assert abs(got_mean - mean_log_prob) < TOLERANCE, f"{got_mean:.6f} is not {mean_log_prob:.6f}"
    print(
        f"forge checkpoint, mean per-residue log-probability: {got_mean:.6f} "
        f"(computed directly: {mean_log_prob:.6f}; tolerance {TOLERANCE})"
    )


if __name__ == "__main__":
    main()
