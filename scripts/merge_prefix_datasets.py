from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUTS = [
    "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_prefix_dataset.csv",
    "outputs/agentdojo_mini_batch_round2/merged/workspace_mini_batch_round2_prefix_dataset.csv",
]
DEFAULT_BATCHES = ["round1", "round2"]
DEFAULT_OUT = "outputs/agentdojo_combined/workspace_combined_prefix_dataset.csv"


def _read_with_source(path: str | Path, source_batch: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing prefix dataset: {path}")
    df = pd.read_csv(path)
    if "episode_id" not in df.columns:
        raise ValueError(f"{path} has no episode_id column.")
    if "hook_type" not in df.columns:
        raise ValueError(f"{path} has no hook_type column.")
    if "source_batch" in df.columns:
        df = df.drop(columns=["source_batch"])
    df.insert(0, "source_batch", source_batch)
    return df


def merge_prefix_datasets(
    inputs: list[str | Path] | None = None,
    source_batches: list[str] | None = None,
    out_path: str | Path = DEFAULT_OUT,
) -> dict[str, Any]:
    inputs = list(inputs or DEFAULT_INPUTS)
    source_batches = list(source_batches or DEFAULT_BATCHES)
    if len(inputs) != len(source_batches):
        raise ValueError("Number of inputs must match number of source batch labels.")

    frames = [_read_with_source(path, batch) for path, batch in zip(inputs, source_batches)]
    merged = pd.concat(frames, ignore_index=True, sort=False)
    input_rows = int(len(merged))
    merged = merged.drop_duplicates(ignore_index=True)
    duplicate_rows_dropped = input_rows - int(len(merged))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    return {
        "status": "ok",
        "inputs": [str(path) for path in inputs],
        "source_batches": source_batches,
        "output": str(out_path),
        "input_rows": input_rows,
        "output_rows": int(len(merged)),
        "duplicate_rows_dropped": duplicate_rows_dropped,
        "episode_count": int(merged["episode_id"].nunique()),
        "source_batch_counts": {
            str(key): int(value)
            for key, value in merged["source_batch"].value_counts().sort_index().items()
        },
        "columns": list(merged.columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge AgentDojo prefix datasets with source batch labels.")
    parser.add_argument("--input", action="append", dest="inputs", default=None)
    parser.add_argument("--source-batch", action="append", dest="source_batches", default=None)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    result = merge_prefix_datasets(
        inputs=args.inputs,
        source_batches=args.source_batches,
        out_path=args.out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
