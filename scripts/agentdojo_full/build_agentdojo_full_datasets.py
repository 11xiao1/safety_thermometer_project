from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.split_prefix_dataset import split_prefix_dataset  # noqa: E402


DEFAULT_BASE_DISAGREEMENT = (
    "outputs/agentdojo_multisuite_disagreement/"
    "agentdojo_multisuite_disagreement_prefix_dataset.csv"
)
DEFAULT_AUDIT = "outputs/agentdojo_expansion_outcome_audit.json"
DEFAULT_FULL_OUT = "outputs/agentdojo_full_combined/agentdojo_full_prefix_dataset.csv"
DEFAULT_FULL_DISAGREEMENT_OUT = (
    "outputs/agentdojo_full_disagreement/"
    "agentdojo_full_disagreement_prefix_dataset.csv"
)
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
SEMANTIC_KEY = ["episode_id", "step_id", "hook_type"]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _task_key(suite: str, task_id: str) -> str:
    return f"{suite}:{task_id}"


def _load_summary_task_results(row: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    payload = _load_json(row["summary_path"])
    suite = str(payload.get("suite", row.get("suite", "")))
    results = {}
    for result in payload.get("task_results", []):
        task_id = result.get("task_id")
        if not task_id:
            continue
        enriched = dict(result)
        enriched["suite"] = suite
        enriched["source_batch"] = row["batch"]
        enriched["source_group"] = row["source_group"]
        results[(suite, str(task_id))] = enriched
    return results


def _effective_expansion_results(audit: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    primary: dict[tuple[str, str], dict[str, Any]] = {}
    recovery: dict[tuple[str, str], dict[str, Any]] = {}
    for row in audit.get("batches", []):
        primary.update(_load_summary_task_results(row))
    for row in audit.get("recovery_batches", []):
        recovery.update(_load_summary_task_results(row))

    effective = {}
    for key, result in primary.items():
        if result.get("status") == "stopped" and key in recovery:
            effective[key] = recovery[key]
        else:
            effective[key] = result
    return effective


def _episode_id_for_task(suite: str, task_id: str) -> str:
    return f"{suite}:{task_id}:none:none"


def _read_prefix_with_metadata(result: dict[str, Any]) -> pd.DataFrame:
    prefix = Path(str(result.get("prefix", "")))
    if not prefix.exists():
        raise FileNotFoundError(f"Missing completed prefix dataset: {prefix}")
    df = pd.read_csv(prefix)
    suite = str(result["suite"])
    task_id = str(result["task_id"])
    source_batch = str(result["source_batch"])
    for column in ["source_suite", "source_batch", "utility", "security"]:
        if column in df.columns:
            df = df.drop(columns=[column])
    df.insert(0, "source_batch", source_batch)
    df.insert(0, "source_suite", suite)
    df["utility"] = result.get("utility")
    df["security"] = result.get("security")
    expected_episode = _episode_id_for_task(suite, task_id)
    if "episode_id" not in df.columns:
        raise ValueError(f"{prefix} has no episode_id column.")
    episode_ids = set(str(value) for value in df["episode_id"].dropna().unique())
    if expected_episode not in episode_ids:
        raise ValueError(f"{prefix} does not contain expected episode_id {expected_episode}.")
    return df


def _base_dataframe(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in [*SEMANTIC_KEY, *STANDARD_COLUMNS, *DISAGREEMENT_COLUMNS] if column not in df.columns]
    if missing:
        raise ValueError("Base disagreement dataset is missing columns: " + ", ".join(missing))
    if "utility" not in df.columns:
        df["utility"] = pd.NA
    if "security" not in df.columns:
        df["security"] = pd.NA
    return df


def _deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicate_rows = int(df.duplicated(subset=SEMANTIC_KEY, keep=False).sum())
    if duplicate_rows:
        rank = df["source_batch"].astype(str).str.contains("recovery", case=False, na=False).astype(int)
        ranked = df.assign(_recovery_rank=rank, _input_order=range(len(df)))
        ranked = ranked.sort_values(
            by=SEMANTIC_KEY + ["_recovery_rank", "_input_order"],
            ascending=[True, True, True, False, True],
            kind="stable",
        )
        ranked = ranked.drop_duplicates(subset=SEMANTIC_KEY, keep="first")
        df = ranked.sort_values("_input_order", kind="stable").drop(columns=["_recovery_rank", "_input_order"])
    return df.reset_index(drop=True), duplicate_rows


def _drop_disagreement_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[column for column in DISAGREEMENT_COLUMNS if column in df.columns])


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = Counter("" if pd.isna(value) else str(value) for value in df[column].tolist())
    return dict(sorted(counts.items()))


def _split_overlap_check(manifest: dict[str, Any]) -> dict[str, Any]:
    episode_ids = {
        split: set(str(value) for value in manifest.get("episode_ids", {}).get(split, []))
        for split in ["train", "val", "test"]
    }
    overlaps = {
        "train_val": sorted(episode_ids["train"] & episode_ids["val"]),
        "train_test": sorted(episode_ids["train"] & episode_ids["test"]),
        "val_test": sorted(episode_ids["val"] & episode_ids["test"]),
    }
    return {
        "passed": not any(overlaps.values()),
        "overlaps": overlaps,
    }


def _episode_source_batch_conflicts(df: pd.DataFrame) -> dict[str, list[str]]:
    conflicts = {}
    if "source_batch" not in df.columns:
        return conflicts
    grouped = df.groupby("episode_id")["source_batch"].nunique()
    conflict_ids = sorted(str(idx) for idx, count in grouped.items() if int(count) > 1)
    for episode_id in conflict_ids:
        values = sorted(str(value) for value in df.loc[df["episode_id"].astype(str) == episode_id, "source_batch"].dropna().unique())
        conflicts[episode_id] = values
    return conflicts


def _augment_manifest(
    manifest: dict[str, Any],
    df: pd.DataFrame,
    excluded_budget_limited_tasks: list[str],
    semantic_duplicate_rows: int,
) -> dict[str, Any]:
    manifest = dict(manifest)
    manifest["total_included_rows"] = int(len(df))
    manifest["total_included_episodes"] = int(df["episode_id"].nunique())
    manifest["excluded_budget_limited_tasks"] = excluded_budget_limited_tasks
    manifest["semantic_duplicate_row_count_after_deduplication"] = int(
        df.duplicated(subset=SEMANTIC_KEY, keep=False).sum()
    )
    manifest["semantic_duplicate_row_count_before_deduplication"] = int(semantic_duplicate_rows)
    manifest["episode_overlap_check"] = _split_overlap_check(manifest)
    manifest["episode_source_batch_conflicts"] = _episode_source_batch_conflicts(df)
    manifest["utility_counts"] = {}
    manifest["security_counts"] = {}

    for split_name, episodes in manifest["episode_ids"].items():
        split_df = df[df["episode_id"].astype(str).isin([str(value) for value in episodes])].copy()
        manifest["utility_counts"][split_name] = _value_counts(split_df, "utility")
        manifest["security_counts"][split_name] = _value_counts(split_df, "security")

    warnings = list(manifest.get("warnings", []))
    if manifest["episode_source_batch_conflicts"]:
        warnings.append("Some episode_id values appear in multiple source_batch values.")
    if not manifest["episode_overlap_check"]["passed"]:
        warnings.append("Train/val/test episode overlap detected.")
    manifest["warnings"] = warnings
    manifest_path = Path(manifest["outputs"]["manifest"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _write_dataset_and_split(
    df: pd.DataFrame,
    out_path: str | Path,
    split_dir: str | Path,
    seed: int,
    excluded_budget_limited_tasks: list[str],
    semantic_duplicate_rows: int,
    prior_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    manifest = split_prefix_dataset(
        input_path=out_path,
        out_dir=split_dir,
        seed=seed,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        prior_manifest_path=prior_manifest_path,
    )
    return _augment_manifest(manifest, df, excluded_budget_limited_tasks, semantic_duplicate_rows)


def build_agentdojo_full_datasets(
    base_disagreement_path: str | Path = DEFAULT_BASE_DISAGREEMENT,
    audit_path: str | Path = DEFAULT_AUDIT,
    full_out: str | Path = DEFAULT_FULL_OUT,
    full_disagreement_out: str | Path = DEFAULT_FULL_DISAGREEMENT_OUT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    audit = _load_json(audit_path)
    budget_limited_tasks = sorted(
        str(value)
        for value in audit.get("effective_after_recovery", {}).get("budget_limited_tasks", [])
    )
    budget_limited_set = set(budget_limited_tasks)
    base_df = _base_dataframe(base_disagreement_path)
    effective = _effective_expansion_results(audit)
    completed_frames = []
    excluded_non_completed = []
    for (suite, task_id), result in sorted(effective.items()):
        full_task = _task_key(suite, task_id)
        if full_task in budget_limited_set:
            continue
        if result.get("status") != "ok":
            excluded_non_completed.append(full_task)
            continue
        completed_frames.append(_read_prefix_with_metadata(result))

    combined_disagreement = pd.concat([base_df, *completed_frames], ignore_index=True, sort=False)
    combined_disagreement, duplicate_rows_before = _deduplicate(combined_disagreement)
    semantic_duplicates_after = int(combined_disagreement.duplicated(subset=SEMANTIC_KEY, keep=False).sum())
    if semantic_duplicates_after:
        raise ValueError("Semantic duplicate rows remain after full dataset deduplication.")

    missing_disagreement = [column for column in DISAGREEMENT_COLUMNS if column not in combined_disagreement.columns]
    if missing_disagreement:
        raise ValueError("Full disagreement dataset is missing columns: " + ", ".join(missing_disagreement))

    combined_no_disagreement = _drop_disagreement_columns(combined_disagreement)
    no_manifest = _write_dataset_and_split(
        df=combined_no_disagreement,
        out_path=full_out,
        split_dir=Path(full_out).parent / "splits",
        seed=seed,
        excluded_budget_limited_tasks=budget_limited_tasks,
        semantic_duplicate_rows=duplicate_rows_before,
    )
    disagreement_manifest = _write_dataset_and_split(
        df=combined_disagreement,
        out_path=full_disagreement_out,
        split_dir=Path(full_disagreement_out).parent / "splits",
        seed=seed,
        excluded_budget_limited_tasks=budget_limited_tasks,
        semantic_duplicate_rows=duplicate_rows_before,
        prior_manifest_path=no_manifest["outputs"]["manifest"],
    )

    return {
        "status": "ok",
        "will_call_provider": False,
        "will_run_agentdojo": False,
        "inputs": {
            "base_disagreement": str(base_disagreement_path),
            "audit": str(audit_path),
        },
        "outputs": {
            "no_disagreement": str(full_out),
            "no_disagreement_manifest": no_manifest["outputs"]["manifest"],
            "disagreement": str(full_disagreement_out),
            "disagreement_manifest": disagreement_manifest["outputs"]["manifest"],
        },
        "included_episode_count": int(combined_disagreement["episode_id"].nunique()),
        "included_row_count": int(len(combined_disagreement)),
        "base_episode_count": int(base_df["episode_id"].nunique()),
        "expansion_completed_episode_count": int(sum(frame["episode_id"].nunique() for frame in completed_frames)),
        "excluded_budget_limited_tasks": budget_limited_tasks,
        "excluded_non_completed_tasks": sorted(excluded_non_completed),
        "semantic_duplicate_row_count_before_deduplication": duplicate_rows_before,
        "semantic_duplicate_row_count_after_deduplication": semantic_duplicates_after,
        "no_disagreement_split_manifest": no_manifest,
        "disagreement_split_manifest": disagreement_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full AgentDojo v2 combined datasets from saved outputs only.")
    parser.add_argument("--base-disagreement", default=DEFAULT_BASE_DISAGREEMENT)
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    parser.add_argument("--full-out", default=DEFAULT_FULL_OUT)
    parser.add_argument("--full-disagreement-out", default=DEFAULT_FULL_DISAGREEMENT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    result = build_agentdojo_full_datasets(
        base_disagreement_path=args.base_disagreement,
        audit_path=args.audit,
        full_out=args.full_out,
        full_disagreement_out=args.full_disagreement_out,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
