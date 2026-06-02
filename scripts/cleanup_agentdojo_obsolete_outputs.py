from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLAN_OUT = "outputs/agentdojo_cleanup_plan.json"
SEMANTIC_KEY = ["episode_id", "step_id", "hook_type"]
PRESERVED_CANONICAL_OUTPUTS = [
    "outputs/agentdojo_multisuite_combined",
    "outputs/agentdojo_multisuite_disagreement",
    "outputs/agentdojo_trace_provenance_audit.json",
    "outputs/agentdojo_trace_provenance_audit.csv",
]
PRESERVED_RAW_TRACE_DIRS = [
    "outputs/agentdojo_mini_batch",
    "outputs/agentdojo_mini_batch_round2",
    "outputs/agentdojo_mini_batch_round3",
    "outputs/agentdojo_mini_batch_round4",
    "outputs/agentdojo_mini_batch_slack_round1",
    "outputs/agentdojo_mini_batch_slack_recovery1",
]
CANDIDATE_OBSOLETE_ARTIFACTS = [
    "outputs/agentdojo_multisuite_disagreement_aligned",
    "outputs/agentdojo_multisuite_disagreement_clean",
    "outputs/splits",
    "outputs/agentdojo_split_risk_estimator_metrics.json",
    "outputs/agentdojo_split_risk_estimator_predictions.csv",
    "outputs/agentdojo_smoke_prefix_dataset.csv",
    "outputs/agentdojo_smoke_trace.jsonl",
    "outputs/table3_toy_risk_estimator.csv",
    "outputs/table3_toy_thermometer.csv",
    "outputs/table4_risk_estimator.csv",
    "outputs/table4_toy_thermometer.csv",
    "outputs/toy_risk_estimator_predictions.csv",
    "outputs/toy_thermometer_scores.csv",
    ".pytest_tmp",
]
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
EXPECTED_ROUND1_EPISODES = {
    f"workspace:user_task_{task_id}:none:none"
    for task_id in range(5)
}
PROTECTED_PREFIXES = ["scripts", "tests", "src", "docs", "prompts", "data"]


def _resolve_inside_workspace(path: str | Path) -> Path:
    path = Path(path)
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Refusing to operate outside workspace: {resolved}")
    return resolved


def _relative_posix(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _validate_candidate_path(path: Path) -> None:
    relative = _relative_posix(path)
    top = relative.split("/", 1)[0]
    if top in PROTECTED_PREFIXES:
        raise ValueError(f"Refusing to delete protected project area: {relative}")
    allowed = {_relative_posix(_resolve_inside_workspace(candidate)) for candidate in CANDIDATE_OBSOLETE_ARTIFACTS}
    if relative not in allowed:
        raise ValueError(f"Refusing to delete path not in obsolete candidate list: {relative}")


def _required_paths() -> dict[str, Path]:
    combined = _resolve_inside_workspace("outputs/agentdojo_multisuite_combined")
    disagreement = _resolve_inside_workspace("outputs/agentdojo_multisuite_disagreement")
    return {
        "combined_dir": combined,
        "combined_prefix": combined / "agentdojo_multisuite_combined_prefix_dataset.csv",
        "combined_train": combined / "splits" / "agentdojo_train.csv",
        "combined_val": combined / "splits" / "agentdojo_val.csv",
        "combined_test": combined / "splits" / "agentdojo_test.csv",
        "combined_manifest": combined / "splits" / "split_manifest.json",
        "disagreement_dir": disagreement,
        "disagreement_prefix": disagreement / "agentdojo_multisuite_disagreement_prefix_dataset.csv",
        "disagreement_train": disagreement / "splits" / "agentdojo_train.csv",
        "disagreement_val": disagreement / "splits" / "agentdojo_val.csv",
        "disagreement_test": disagreement / "splits" / "agentdojo_test.csv",
        "disagreement_manifest": disagreement / "splits" / "split_manifest.json",
        "round1_prefix": _resolve_inside_workspace("outputs/agentdojo_mini_batch/merged/workspace_mini_batch_prefix_dataset.csv"),
    }


def _semantic_duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated(subset=SEMANTIC_KEY, keep=False).sum())


def _validate_canonical_outputs() -> dict[str, Any]:
    paths = _required_paths()
    missing_paths = [str(path) for path in paths.values() if not path.exists()]
    errors: list[str] = []
    if missing_paths:
        return {
            "passed": False,
            "errors": ["Canonical required paths are missing."],
            "missing_paths": missing_paths,
        }

    disagreement = pd.read_csv(paths["disagreement_prefix"])
    split_frames = []
    for split_name in ["train", "val", "test"]:
        split_df = pd.read_csv(paths[f"disagreement_{split_name}"])
        split_df["_split"] = split_name
        split_frames.append(split_df)
    splits = pd.concat(split_frames, ignore_index=True)
    round1 = pd.read_csv(paths["round1_prefix"])

    semantic_duplicate_rows = _semantic_duplicate_count(disagreement)
    if semantic_duplicate_rows:
        errors.append("Canonical disagreement combined has semantic duplicate rows.")

    split_overlap_episodes = sorted(
        splits.groupby("episode_id")["_split"].nunique().loc[lambda counts: counts > 1].index.astype(str).tolist()
    )
    if split_overlap_episodes:
        errors.append("Canonical disagreement train/val/test splits overlap by episode_id.")

    missing_required_columns = [
        column for column in REQUIRED_DISAGREEMENT_COLUMNS if column not in disagreement.columns
    ]
    if missing_required_columns:
        errors.append("Canonical disagreement combined is missing disagreement feature columns.")

    round1_episodes = set(round1["episode_id"].astype(str).unique())
    unexpected_round1_episodes = sorted(round1_episodes - EXPECTED_ROUND1_EPISODES)
    missing_round1_episodes = sorted(EXPECTED_ROUND1_EPISODES - round1_episodes)
    if unexpected_round1_episodes or missing_round1_episodes:
        errors.append("Repaired round1 prefix does not contain exactly workspace:user_task_0 through workspace:user_task_4.")

    return {
        "passed": not errors,
        "errors": errors,
        "missing_paths": missing_paths,
        "canonical_disagreement_rows": int(len(disagreement)),
        "canonical_disagreement_episodes": int(disagreement["episode_id"].nunique()),
        "canonical_disagreement_semantic_duplicate_rows": semantic_duplicate_rows,
        "split_overlap_episodes": split_overlap_episodes,
        "missing_required_disagreement_columns": missing_required_columns,
        "round1_episode_ids": sorted(round1_episodes),
        "unexpected_round1_episodes": unexpected_round1_episodes,
        "missing_round1_episodes": missing_round1_episodes,
    }


def cleanup_agentdojo_obsolete_outputs(
    apply: bool = False,
    plan_out: str | Path = PLAN_OUT,
) -> dict[str, Any]:
    validation = _validate_canonical_outputs()
    candidate_paths = [_resolve_inside_workspace(candidate) for candidate in CANDIDATE_OBSOLETE_ARTIFACTS]
    for candidate in candidate_paths:
        _validate_candidate_path(candidate)
    existing_candidates = [path for path in candidate_paths if path.exists()]

    deleted: list[str] = []
    if apply and validation["passed"]:
        for candidate in existing_candidates:
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
            deleted.append(str(candidate))

    plan = {
        "apply": apply,
        "validation": validation,
        "candidate_delete_list": [str(path) for path in candidate_paths],
        "existing_candidates": [str(path) for path in existing_candidates],
        "would_delete": [] if apply else [str(path) for path in existing_candidates],
        "deleted": deleted,
        "preserved_canonical_outputs": [
            str(_resolve_inside_workspace(path)) for path in PRESERVED_CANONICAL_OUTPUTS
        ],
        "preserved_raw_trace_dirs": [
            str(_resolve_inside_workspace(path)) for path in PRESERVED_RAW_TRACE_DIRS
        ],
        "skipped_reason": None if validation["passed"] else "canonical_validation_failed",
    }
    plan_path = _resolve_inside_workspace(plan_out)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate canonical AgentDojo outputs and clean explicitly listed obsolete output artifacts."
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--plan-out", default=PLAN_OUT)
    args = parser.parse_args()

    plan = cleanup_agentdojo_obsolete_outputs(apply=args.apply, plan_out=args.plan_out)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not plan["validation"]["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
