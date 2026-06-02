from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUTS = [
    "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_prefix_dataset.csv",
    "outputs/agentdojo_mini_batch_round2/merged/workspace_mini_batch_round2_prefix_dataset.csv",
    "outputs/agentdojo_mini_batch_round3/merged/workspace_mini_batch_round3_prefix_dataset.csv",
    "outputs/agentdojo_mini_batch_round4/merged/workspace_mini_batch_round4_prefix_dataset.csv",
    "outputs/agentdojo_mini_batch_slack_round1/merged/slack_mini_batch_round1_prefix_dataset.csv",
    "outputs/agentdojo_mini_batch_slack_recovery1/merged/slack_recovery1_prefix_dataset.csv",
]
DEFAULT_BATCHES = [
    "workspace_round1",
    "workspace_round2",
    "workspace_round3",
    "workspace_round4",
    "slack_round1",
    "slack_recovery1",
]
DEFAULT_OUT = "outputs/agentdojo_multisuite_combined/agentdojo_multisuite_combined_prefix_dataset.csv"


def _infer_source_suite(source_batch: str, path: str | Path) -> str:
    text = f"{source_batch} {path}".lower()
    if "slack" in text:
        return "slack"
    if "workspace" in text:
        return "workspace"
    return "unknown"


def _read_with_source(path: str | Path, source_batch: str, source_suite: str | None = None) -> pd.DataFrame:
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
    if "source_suite" in df.columns:
        df = df.drop(columns=["source_suite"])
    source_suite = source_suite or _infer_source_suite(source_batch, path)
    df.insert(0, "source_batch", source_batch)
    df.insert(0, "source_suite", source_suite)
    return df


def _deduplicate_with_recovery_preference(merged: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    input_rows = int(len(merged))
    merged = merged.drop_duplicates(ignore_index=True)
    duplicate_rows_dropped = input_rows - int(len(merged))

    identity_columns = [
        column
        for column in ["episode_id", "step_id", "hook_type"]
        if column in merged.columns
    ]
    if not identity_columns:
        return merged, duplicate_rows_dropped, 0

    recovery_rank = merged["source_batch"].astype(str).str.contains("recovery", case=False, na=False).astype(int)
    ranked = merged.assign(_recovery_rank=recovery_rank, _input_order=range(len(merged)))
    ranked = ranked.sort_values(
        by=identity_columns + ["_recovery_rank", "_input_order"],
        ascending=[True] * len(identity_columns) + [False, True],
        kind="stable",
    )
    group_has_recovery = ranked.groupby(identity_columns)["_recovery_rank"].transform("max").astype(bool)
    conflict_mask = ranked.duplicated(subset=identity_columns, keep="first") & group_has_recovery
    conflict_rows_dropped = int(conflict_mask.sum())
    ranked = ranked[~conflict_mask].sort_values("_input_order", kind="stable")
    ranked = ranked.drop(columns=["_recovery_rank", "_input_order"]).reset_index(drop=True)
    return ranked, duplicate_rows_dropped, conflict_rows_dropped


def merge_prefix_datasets(
    inputs: list[str | Path] | None = None,
    source_batches: list[str] | None = None,
    out_path: str | Path = DEFAULT_OUT,
    source_suites: list[str] | None = None,
) -> dict[str, Any]:
    inputs = list(inputs or DEFAULT_INPUTS)
    source_batches = list(source_batches or DEFAULT_BATCHES)
    if len(inputs) != len(source_batches):
        raise ValueError("Number of inputs must match number of source batch labels.")
    if source_suites is None:
        source_suites = [
            _infer_source_suite(batch, path)
            for path, batch in zip(inputs, source_batches)
        ]
    else:
        source_suites = list(source_suites)
        if len(inputs) != len(source_suites):
            raise ValueError("Number of inputs must match number of source suite labels.")

    frames = [
        _read_with_source(path, batch, suite)
        for path, batch, suite in zip(inputs, source_batches, source_suites)
    ]
    merged = pd.concat(frames, ignore_index=True, sort=False)
    input_rows = int(len(merged))
    merged, duplicate_rows_dropped, conflict_rows_dropped = _deduplicate_with_recovery_preference(merged)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    return {
        "status": "ok",
        "inputs": [str(path) for path in inputs],
        "source_batches": source_batches,
        "source_suites": source_suites,
        "output": str(out_path),
        "input_rows": input_rows,
        "output_rows": int(len(merged)),
        "duplicate_rows_dropped": duplicate_rows_dropped,
        "conflict_rows_dropped": conflict_rows_dropped,
        "episode_count": int(merged["episode_id"].nunique()),
        "source_suite_counts": {
            str(key): int(value)
            for key, value in merged["source_suite"].value_counts().sort_index().items()
        },
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
    parser.add_argument("--source-suite", action="append", dest="source_suites", default=None)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    result = merge_prefix_datasets(
        inputs=args.inputs,
        source_batches=args.source_batches,
        source_suites=args.source_suites,
        out_path=args.out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
