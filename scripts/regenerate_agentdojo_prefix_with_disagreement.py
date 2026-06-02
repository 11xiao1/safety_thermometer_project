from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.align_agentdojo_disagreement_split import align_agentdojo_disagreement_split  # noqa: E402
from scripts.merge_prefix_datasets import merge_prefix_datasets  # noqa: E402
from src.monitor.replay import make_prefix_dataset  # noqa: E402


DEFAULT_TRACE_INPUTS = [
    "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_trace.jsonl",
    "outputs/agentdojo_mini_batch_round2/merged/workspace_mini_batch_round2_trace.jsonl",
    "outputs/agentdojo_mini_batch_round3/merged/workspace_mini_batch_round3_trace.jsonl",
    "outputs/agentdojo_mini_batch_round4/merged/workspace_mini_batch_round4_trace.jsonl",
    "outputs/agentdojo_mini_batch_slack_round1/merged/slack_mini_batch_round1_trace.jsonl",
    "outputs/agentdojo_mini_batch_slack_recovery1/merged/slack_recovery1_trace.jsonl",
]
DEFAULT_SOURCE_BATCHES = [
    "workspace_round1",
    "workspace_round2",
    "workspace_round3",
    "workspace_round4",
    "slack_round1",
    "slack_recovery1",
]
DEFAULT_SOURCE_SUITES = ["workspace", "workspace", "workspace", "workspace", "slack", "slack"]
DEFAULT_PREFIX_OUTPUTS = [
    "outputs/agentdojo_multisuite_disagreement/prefix/workspace_round1_prefix_dataset.csv",
    "outputs/agentdojo_multisuite_disagreement/prefix/workspace_round2_prefix_dataset.csv",
    "outputs/agentdojo_multisuite_disagreement/prefix/workspace_round3_prefix_dataset.csv",
    "outputs/agentdojo_multisuite_disagreement/prefix/workspace_round4_prefix_dataset.csv",
    "outputs/agentdojo_multisuite_disagreement/prefix/slack_round1_prefix_dataset.csv",
    "outputs/agentdojo_multisuite_disagreement/prefix/slack_recovery1_prefix_dataset.csv",
]
DEFAULT_COMBINED_OUT = (
    "outputs/agentdojo_multisuite_disagreement/"
    "agentdojo_multisuite_disagreement_prefix_dataset.csv"
)
DEFAULT_SPLIT_DIR = "outputs/agentdojo_multisuite_disagreement/splits"
DEFAULT_PRIOR_MANIFEST = "outputs/agentdojo_multisuite_combined/splits/split_manifest.json"
DEFAULT_SEED = 42
DISAGREEMENT_COLUMNS = [
    "f_intent_tool_mismatch",
    "f_plan_action_mismatch",
    "f_sensitive_tool_without_need",
    "f_external_send_without_request",
    "f_observation_error",
    "f_self_check_risk_mismatch",
    "f_fallback_after_error",
    "cumulative_observation_error_count",
    "cumulative_disagreement_count",
    "max_disagreement_score_so_far",
]
STANDARD_COLUMNS = [
    "future_risk_label",
    "oracle_violation",
    "risk_score",
    "max_risk_score_so_far",
]


def _validate_lengths(*items: list[Any]) -> None:
    lengths = {len(item) for item in items}
    if len(lengths) != 1:
        raise ValueError("Trace inputs, outputs, source batches, and source suites must have equal lengths.")


def _trusted_trace_path_for_prefix(out_path: Path) -> Path:
    stem = out_path.stem.replace("_prefix_dataset", "")
    return out_path.parent.parent / "trusted_traces" / f"{stem}_trace.jsonl"


def _materialize_trace_source(trace_path: Path, out_path: Path) -> tuple[Path, dict[str, Any]]:
    if trace_path.is_file():
        return trace_path, {
            "trace_source_kind": "file",
            "trusted_trace": str(trace_path),
            "source_trace_file_count": 1,
        }
    if not trace_path.is_dir():
        raise FileNotFoundError(f"Missing trace source: {trace_path}")

    trace_files = sorted(trace_path.glob("*.jsonl"))
    if not trace_files:
        raise FileNotFoundError(f"Trace directory contains no JSONL files: {trace_path}")

    trusted_trace = _trusted_trace_path_for_prefix(out_path)
    trusted_trace.parent.mkdir(parents=True, exist_ok=True)
    with trusted_trace.open("w", encoding="utf-8") as out_file:
        for trace_file in trace_files:
            text = trace_file.read_text(encoding="utf-8").strip()
            if text:
                out_file.write(text)
                out_file.write("\n")
    return trusted_trace, {
        "trace_source_kind": "directory",
        "trusted_trace": str(trusted_trace),
        "source_trace_file_count": len(trace_files),
        "source_trace_files": [str(trace_file) for trace_file in trace_files],
    }


def _write_prefix_from_trace(
    trace_path: str | Path,
    out_path: str | Path,
    source_suite: str,
    source_batch: str,
) -> dict[str, Any]:
    trace_path = Path(trace_path)
    out_path = Path(out_path)
    trusted_trace_path, source_info = _materialize_trace_source(trace_path, out_path)

    df = make_prefix_dataset(str(trusted_trace_path))
    missing_columns = [
        column
        for column in [*DISAGREEMENT_COLUMNS, *STANDARD_COLUMNS]
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Regenerated prefix dataset is missing required columns: {missing_columns}")

    for column in ["source_batch", "source_suite"]:
        if column in df.columns:
            df = df.drop(columns=[column])
    df.insert(0, "source_batch", source_batch)
    df.insert(0, "source_suite", source_suite)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return {
        "trace": str(trace_path),
        **source_info,
        "output": str(out_path),
        "source_suite": source_suite,
        "source_batch": source_batch,
        "rows": int(len(df)),
        "episodes": int(df["episode_id"].nunique()) if "episode_id" in df.columns else 0,
    }


def regenerate_agentdojo_prefix_with_disagreement(
    trace_inputs: list[str | Path] | None = None,
    prefix_outputs: list[str | Path] | None = None,
    source_batches: list[str] | None = None,
    source_suites: list[str] | None = None,
    combined_out: str | Path = DEFAULT_COMBINED_OUT,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    prior_manifest_path: str | Path | None = DEFAULT_PRIOR_MANIFEST,
    seed: int = DEFAULT_SEED,
    strict_align_splits: bool = True,
) -> dict[str, Any]:
    trace_inputs = list(trace_inputs or DEFAULT_TRACE_INPUTS)
    prefix_outputs = list(prefix_outputs or DEFAULT_PREFIX_OUTPUTS)
    source_batches = list(source_batches or DEFAULT_SOURCE_BATCHES)
    source_suites = list(source_suites or DEFAULT_SOURCE_SUITES)
    _validate_lengths(trace_inputs, prefix_outputs, source_batches, source_suites)

    prefix_results = [
        _write_prefix_from_trace(trace_path, out_path, suite, batch)
        for trace_path, out_path, suite, batch in zip(
            trace_inputs,
            prefix_outputs,
            source_suites,
            source_batches,
        )
    ]
    merge_result = merge_prefix_datasets(
        inputs=prefix_outputs,
        source_batches=source_batches,
        out_path=combined_out,
        source_suites=source_suites,
    )
    combined = pd.read_csv(combined_out)
    semantic_key = ["episode_id", "step_id", "hook_type"]
    duplicate_mask = combined.duplicated(subset=semantic_key, keep=False)
    semantic_duplicate_rows = int(duplicate_mask.sum())
    if semantic_duplicate_rows:
        duplicate_examples = combined.loc[duplicate_mask, semantic_key + ["source_batch"]].head(20).to_dict("records")
        raise ValueError(
            "Semantic duplicate prefix rows remain after regeneration: "
            + json.dumps(duplicate_examples, sort_keys=True)
        )
    if strict_align_splits:
        split_manifest = align_agentdojo_disagreement_split(
            input_path=combined_out,
            prior_manifest_path=prior_manifest_path,
            out_dir=split_dir,
        )
    else:
        from scripts.split_prefix_dataset import split_prefix_dataset

        split_manifest = split_prefix_dataset(
            input_path=combined_out,
            out_dir=split_dir,
            seed=seed,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            prior_manifest_path=prior_manifest_path,
        )
    return {
        "status": "ok",
        "prefix_outputs": prefix_results,
        "combined_output": str(combined_out),
        "split_dir": str(split_dir),
        "required_disagreement_columns": DISAGREEMENT_COLUMNS,
        "combined_row_count": int(len(combined)),
        "combined_episode_count": int(combined["episode_id"].nunique()),
        "semantic_duplicate_rows_after_merge": semantic_duplicate_rows,
        "strict_align_splits": strict_align_splits,
        "source_suite_counts": {
            str(key): int(value)
            for key, value in combined["source_suite"].value_counts().sort_index().items()
        },
        "source_batch_counts": {
            str(key): int(value)
            for key, value in combined["source_batch"].value_counts().sort_index().items()
        },
        "merge": merge_result,
        "split_manifest": split_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate AgentDojo prefix datasets with disagreement features from saved traces only."
    )
    parser.add_argument("--trace", action="append", dest="trace_inputs", default=None)
    parser.add_argument("--trace-source", action="append", dest="trace_inputs", default=None)
    parser.add_argument("--prefix-out", action="append", dest="prefix_outputs", default=None)
    parser.add_argument("--source-batch", action="append", dest="source_batches", default=None)
    parser.add_argument("--source-suite", action="append", dest="source_suites", default=None)
    parser.add_argument("--combined-out", default=DEFAULT_COMBINED_OUT)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--reuse-manifest", default=DEFAULT_PRIOR_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--random-resplit", action="store_true")
    args = parser.parse_args()

    result = regenerate_agentdojo_prefix_with_disagreement(
        trace_inputs=args.trace_inputs,
        prefix_outputs=args.prefix_outputs,
        source_batches=args.source_batches,
        source_suites=args.source_suites,
        combined_out=args.combined_out,
        split_dir=args.split_dir,
        prior_manifest_path=args.reuse_manifest,
        seed=args.seed,
        strict_align_splits=not args.random_resplit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
