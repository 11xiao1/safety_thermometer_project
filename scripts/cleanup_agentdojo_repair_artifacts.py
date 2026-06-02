from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP = "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_trace.corrupt_backup.jsonl"
DEFAULT_CLEAN_DIR = "outputs/agentdojo_multisuite_disagreement_clean"
ROUND1_TRACE = "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_trace.jsonl"
ROUND1_PREFIX = "outputs/agentdojo_mini_batch/merged/workspace_mini_batch_prefix_dataset.csv"
CANONICAL_COMBINED = "outputs/agentdojo_multisuite_disagreement/agentdojo_multisuite_disagreement_prefix_dataset.csv"
CANONICAL_SPLIT_DIR = "outputs/agentdojo_multisuite_disagreement/splits"
EXPECTED_ROUND1_EPISODES = {
    f"workspace:user_task_{task_id}:none:none"
    for task_id in range(5)
}
SEMANTIC_KEY = ["episode_id", "step_id", "hook_type"]
REQUIRED_DISAGREEMENT_COLUMNS = [
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
REQUIRED_STANDARD_COLUMNS = [
    "future_risk_label",
    "oracle_violation",
    "risk_score",
    "max_risk_score_so_far",
]


def _resolve_inside_workspace(path: str | Path) -> Path:
    resolved = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Refusing to operate outside workspace: {resolved}")
    return resolved


def _canonical_paths() -> dict[str, Path]:
    split_dir = _resolve_inside_workspace(CANONICAL_SPLIT_DIR)
    return {
        "round1_trace": _resolve_inside_workspace(ROUND1_TRACE),
        "round1_prefix": _resolve_inside_workspace(ROUND1_PREFIX),
        "canonical_combined": _resolve_inside_workspace(CANONICAL_COMBINED),
        "train": split_dir / "agentdojo_train.csv",
        "val": split_dir / "agentdojo_val.csv",
        "test": split_dir / "agentdojo_test.csv",
        "manifest": split_dir / "split_manifest.json",
    }


def _semantic_duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated(subset=SEMANTIC_KEY, keep=False).sum())


def _validate_canonical_outputs() -> dict[str, Any]:
    paths = _canonical_paths()
    missing_paths = [str(path) for path in paths.values() if not path.exists()]
    if missing_paths:
        return {
            "passed": False,
            "missing_paths": missing_paths,
            "errors": ["Canonical outputs are missing."],
        }

    errors: list[str] = []
    round1_prefix = pd.read_csv(paths["round1_prefix"])
    combined = pd.read_csv(paths["canonical_combined"])
    split_frames = []
    for split_name in ["train", "val", "test"]:
        split_df = pd.read_csv(paths[split_name])
        split_df["_split"] = split_name
        split_frames.append(split_df)
    splits = pd.concat(split_frames, ignore_index=True)

    round1_episodes = set(round1_prefix["episode_id"].astype(str).unique())
    unexpected_round1_episodes = sorted(round1_episodes - EXPECTED_ROUND1_EPISODES)
    missing_round1_episodes = sorted(EXPECTED_ROUND1_EPISODES - round1_episodes)
    if unexpected_round1_episodes or missing_round1_episodes:
        errors.append("Round1 prefix episodes do not match workspace:user_task_0 through workspace:user_task_4.")

    round1_semantic_duplicates = _semantic_duplicate_count(round1_prefix)
    if round1_semantic_duplicates:
        errors.append("Round1 prefix has semantic duplicate rows.")

    combined_semantic_duplicates = _semantic_duplicate_count(combined)
    if combined_semantic_duplicates:
        errors.append("Canonical combined disagreement dataset has semantic duplicate rows.")

    multi_source_batch_episodes = []
    if "source_batch" in combined.columns:
        source_batch_counts = combined.groupby("episode_id")["source_batch"].nunique()
        multi_source_batch_episodes = sorted(source_batch_counts[source_batch_counts > 1].index.astype(str).tolist())
        if multi_source_batch_episodes:
            errors.append("Some episode_ids appear in more than one source_batch.")
    else:
        errors.append("Canonical combined disagreement dataset is missing source_batch.")

    split_overlap_episodes = sorted(
        splits.groupby("episode_id")["_split"].nunique().loc[lambda counts: counts > 1].index.astype(str).tolist()
    )
    if split_overlap_episodes:
        errors.append("Some episode_ids appear in more than one train/val/test split.")

    missing_required_columns = [
        column
        for column in [*REQUIRED_DISAGREEMENT_COLUMNS, *REQUIRED_STANDARD_COLUMNS]
        if column not in combined.columns
    ]
    if missing_required_columns:
        errors.append("Canonical combined disagreement dataset is missing required columns.")

    return {
        "passed": not errors,
        "errors": errors,
        "missing_paths": missing_paths,
        "round1_episode_ids": sorted(round1_episodes),
        "unexpected_round1_episodes": unexpected_round1_episodes,
        "missing_round1_episodes": missing_round1_episodes,
        "round1_semantic_duplicate_rows": round1_semantic_duplicates,
        "combined_row_count": int(len(combined)),
        "combined_episode_count": int(combined["episode_id"].nunique()),
        "combined_semantic_duplicate_rows": combined_semantic_duplicates,
        "multi_source_batch_episodes": multi_source_batch_episodes,
        "split_overlap_episodes": split_overlap_episodes,
        "missing_required_columns": missing_required_columns,
        "split_row_counts": {
            split_name: int(count)
            for split_name, count in splits.groupby("_split").size().to_dict().items()
        },
        "split_episode_counts": {
            split_name: int(count)
            for split_name, count in splits.groupby("_split")["episode_id"].nunique().to_dict().items()
        },
    }


def cleanup_agentdojo_repair_artifacts(
    apply: bool = False,
    backup_path: str | Path = DEFAULT_BACKUP,
    clean_dir: str | Path = DEFAULT_CLEAN_DIR,
) -> dict[str, Any]:
    validation = _validate_canonical_outputs()
    targets = [
        _resolve_inside_workspace(backup_path),
        _resolve_inside_workspace(clean_dir),
    ]
    existing_targets = [target for target in targets if target.exists()]
    deleted: list[str] = []
    if apply and validation["passed"]:
        for target in existing_targets:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            deleted.append(str(target))

    return {
        "apply": apply,
        "validation": validation,
        "targets": [str(target) for target in targets],
        "existing_targets": [str(target) for target in existing_targets],
        "would_delete": [str(target) for target in existing_targets] if not apply else [],
        "deleted": deleted,
        "skipped_reason": None if validation["passed"] else "canonical_validation_failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate canonical AgentDojo disagreement outputs and clean temporary repair artifacts."
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete temporary repair artifacts.")
    parser.add_argument("--backup-path", default=DEFAULT_BACKUP)
    parser.add_argument("--clean-dir", default=DEFAULT_CLEAN_DIR)
    args = parser.parse_args()

    result = cleanup_agentdojo_repair_artifacts(
        apply=args.apply,
        backup_path=args.backup_path,
        clean_dir=args.clean_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["validation"]["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
