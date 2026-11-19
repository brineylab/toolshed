"""Compare the CLI's per-position results with the reference.

The reference was computed once with AbLM-Eval's Python API (evaluate_ablms)
at commit 641cff8, on the same fixture, model and pinned versions this image
carries (test/reference.json); the CLI must reproduce it to 1e-4.

Usage: check.py RESULTS_DIR
"""

import json
import platform
import sys
from pathlib import Path

import pandas as pd

TOLERANCE = 1e-4


def main() -> None:
    results_dir = Path(sys.argv[1])
    reference = json.loads(Path("/toolshed/test/reference.json").read_text())

    (result_file,) = results_dir.rglob("*per-position-inference.parquet")
    df = pd.read_parquet(result_file)

    assert len(df) == reference["n_sequences"], f"{len(df)} sequences, not {reference['n_sequences']}"
    separator = reference["separator"]
    assert df["sequence"].str.contains(separator, regex=False).all(), (
        f"the chain separator {separator!r} is missing from the tokenized sequences"
    )
    token_ids = [int(t) for t in df["tokenized_sequence"].iloc[0]]
    assert token_ids == reference["first_token_ids"], (
        f"the first paired input's token ids {token_ids} differ from the reference "
        f"{reference['first_token_ids']}: the separator or tokenizer changed"
    )

    losses = [x for row in df["loss"] for x in row]
    mean_log_prob = -(sum(losses) / len(losses))
    per_sequence = [-sum(row) / len(row) for row in df["loss"]]

    assert abs(mean_log_prob - reference["mean_log_prob"]) < TOLERANCE, (
        f"mean per-residue log-probability {mean_log_prob:.6f} differs from the "
        f"reference {reference['mean_log_prob']:.6f} by more than {TOLERANCE}"
    )
    for got, want in zip(per_sequence, reference["per_sequence_mean_log_prob"]):
        assert abs(got - want) < TOLERANCE, (
            f"a sequence's mean per-residue log-probability {got:.6f} differs from "
            f"the reference {want:.6f} by more than {TOLERANCE}"
        )

    print("python", platform.python_version())
    print("token ids of the first paired input:", df["tokenized_sequence"].iloc[0])
    print(
        f"mean per-residue log-probability: {mean_log_prob:.6f} "
        f"(reference {reference['mean_log_prob']:.6f}, tolerance {TOLERANCE})"
    )


if __name__ == "__main__":
    main()
