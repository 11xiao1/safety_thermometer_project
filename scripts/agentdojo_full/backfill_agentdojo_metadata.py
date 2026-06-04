from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_FULL_COMBINED = "outputs/agentdojo_full_combined/agentdojo_full_prefix_dataset.csv"
DEFAULT_FULL_DISAGREEMENT = "outputs/agentdojo_full_disagreement/agentdojo_full_disagreement_prefix_dataset.csv"
DEFAULT_COMBINED_SPLIT_DIR = "outputs/agentdojo_full_combined/splits"
DEFAULT_DISAGREEMENT_SPLIT_DIR = "outputs/agentdojo_full_disagreement/splits"
DEFAULT_AUDIT = "outputs/agentdojo_expansion_outcome_audit.json"
DEFAULT_JSON_REPORT = "outputs/agentdojo_metadata_backfill_report.json"
DEFAULT_CSV_REPORT = "outputs/agentdojo_metadata_backfill_report.csv"
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
DATASET_PATHS = [
    DEFAULT_FULL_COMBINED,
    f"{DEFAULT_COMBINED_SPLIT_DIR}/agentdojo_train.csv",
    f"{DEFAULT_COMBINED_SPLIT_DIR}/agentdojo_val.csv",
    f"{DEFAULT_COMBINED_SPLIT_DIR}/agentdojo_test.csv",
    DEFAULT_FULL_DISAGREEMENT,
    f"{DEFAULT_DISAGREEMENT_SPLIT_DIR}/agentdojo_train.csv",
    f"{DEFAULT_DISAGREEMENT_SPLIT_DIR}/agentdojo_val.csv",
    f"{DEFAULT_DISAGREEMENT_SPLIT_DIR}/agentdojo_test.csv",
]
SUMMARY_GLOBS = [
    "outputs/agentdojo_mini_batch*/run_summary.json",
    "outputs/agentdojo_mini_batch_slack_round1/run_summary.json",
    "outputs/agentdojo_mini_batch_slack_recovery1/run_summary.json",
    "outputs/agentdojo_expansion/*/run_summary.json",
    "outputs/agentdojo_expansion_recovery/*/run_summary.json",
]
REPORT_COLUMNS = [
    "episode_id",
    "metadata_status",
    "utility",
    "security",
    "metadata_source_batch",
    "source_count",
    "conflict_details",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _episode_id(suite: str, task_id: str) -> str:
    return f"{suite}:{task_id}:none:none"


def _normalize_bool(value: Any) -> bool | None:
    if value is True or value is False:
        return bool(value)
    if value is None or pd.isna(value):
        return None
    if value == "":
        return None
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _status_for_result(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return "unknown_not_evaluated"
    if "utility" not in result and "security" not in result:
        return "unknown_not_evaluated"
    return "known"


def _source_batch_from_summary(path: Path) -> str:
    return path.parent.name


def _collect_run_summaries(patterns: list[str] | None = None) -> list[Path]:
    patterns = patterns or SUMMARY_GLOBS
    paths: set[Path] = set()
    for pattern in patterns:
        path = Path(pattern)
        if path.is_absolute():
            if path.exists():
                paths.add(path)
            continue
        paths.update(Path(".").glob(pattern))
    return sorted(path for path in paths if path.exists())


def _build_metadata_map(
    summary_paths: list[Path],
    budget_limited_tasks: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    coverage_by_batch: Counter[str] = Counter()
    for summary_path in summary_paths:
        payload = _load_json(summary_path)
        suite = str(payload.get("suite", ""))
        source_batch = _source_batch_from_summary(summary_path)
        for result in payload.get("task_results", []):
            task_id = result.get("task_id")
            if not suite or not task_id:
                continue
            episode_id = _episode_id(suite, str(task_id))
            full_task = f"{suite}:{task_id}"
            status = _status_for_result(result)
            if full_task in budget_limited_tasks:
                status = "unknown_not_evaluated"
            utility = _normalize_bool(result.get("utility"))
            security = _normalize_bool(result.get("security"))
            sources[episode_id].append({
                "summary": str(summary_path),
                "source_batch": source_batch,
                "task_status": result.get("status"),
                "metadata_status": status,
                "utility": utility,
                "security": security,
            })
            coverage_by_batch[source_batch] += 1

    metadata: dict[str, dict[str, Any]] = {}
    for episode_id, episode_sources in sources.items():
        known_sources = [source for source in episode_sources if source["metadata_status"] == "known"]
        utility_values = {source["utility"] for source in known_sources if source["utility"] is not None}
        security_values = {source["security"] for source in known_sources if source["security"] is not None}
        if len(utility_values) > 1 or len(security_values) > 1:
            metadata[episode_id] = {
                "metadata_status": "conflict",
                "utility": None,
                "security": None,
                "metadata_source_batch": ";".join(sorted({source["source_batch"] for source in episode_sources})),
                "sources": episode_sources,
                "conflict_details": {
                    "utility_values": sorted(str(value) for value in utility_values),
                    "security_values": sorted(str(value) for value in security_values),
                },
            }
            continue
        if known_sources:
            utility = next(iter(utility_values)) if utility_values else None
            security = next(iter(security_values)) if security_values else None
            metadata[episode_id] = {
                "metadata_status": "known_false" if utility is False or security is False else "known_true",
                "utility": utility,
                "security": security,
                "metadata_source_batch": ";".join(sorted({source["source_batch"] for source in known_sources})),
                "sources": episode_sources,
                "conflict_details": {},
            }
            continue
        metadata[episode_id] = {
            "metadata_status": "unknown_not_evaluated",
            "utility": None,
            "security": None,
            "metadata_source_batch": ";".join(sorted({source["source_batch"] for source in episode_sources})),
            "sources": episode_sources,
            "conflict_details": {},
        }
    return metadata, dict(sorted(coverage_by_batch.items()))


def _status_bucket(value: Any, metadata_status: str) -> str:
    normalized = _normalize_bool(value)
    if normalized is True:
        return "true"
    if normalized is False:
        return "false"
    if metadata_status in {"unknown_no_summary", "unknown_not_evaluated", "conflict"}:
        return metadata_status
    return "unknown_no_summary"


def _counts_for_split(df: pd.DataFrame, column: str) -> dict[str, int]:
    counts = Counter()
    for _, row in df.iterrows():
        counts[_status_bucket(row.get(column), str(row.get("metadata_status", "")))] += 1
    return {key: int(counts.get(key, 0)) for key in ["true", "false", "unknown_no_summary", "unknown_not_evaluated", "conflict"]}


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
    return {"passed": not any(overlaps.values()), "overlaps": overlaps}


def _apply_metadata(df: pd.DataFrame, metadata: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.copy()
    before_utility = df["utility"].map(_normalize_bool) if "utility" in df.columns else pd.Series([None] * len(df))
    before_security = df["security"].map(_normalize_bool) if "security" in df.columns else pd.Series([None] * len(df))
    for column in ["utility", "security", "metadata_source_batch", "metadata_status"]:
        if column not in df.columns:
            df[column] = pd.NA

    statuses = []
    source_batches = []
    utilities = []
    securities = []
    for _, row in df.iterrows():
        episode_id = str(row["episode_id"])
        item = metadata.get(episode_id)
        if item is None:
            statuses.append("unknown_no_summary")
            source_batches.append("")
            utilities.append(pd.NA)
            securities.append(pd.NA)
            continue
        statuses.append(item["metadata_status"])
        source_batches.append(item["metadata_source_batch"])
        utilities.append(item["utility"] if item["utility"] is not None else pd.NA)
        securities.append(item["security"] if item["security"] is not None else pd.NA)
    df["metadata_status"] = statuses
    df["metadata_source_batch"] = source_batches
    df["utility"] = utilities
    df["security"] = securities

    after_utility = df["utility"].map(_normalize_bool)
    after_security = df["security"].map(_normalize_bool)
    row_backfilled_mask = (
        (before_utility.isna() & after_utility.notna())
        | (before_security.isna() & after_security.notna())
    )
    episode_backfilled = set(df.loc[row_backfilled_mask, "episode_id"].astype(str).tolist())
    return df, {
        "rows_backfilled": int(row_backfilled_mask.sum()),
        "episodes_backfilled": len(episode_backfilled),
    }


def _update_manifest(
    manifest_path: str | Path,
    dataset_path: str | Path,
    metadata_report: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = _load_json(manifest_path)
    dataset = pd.read_csv(dataset_path)
    manifest["metadata_backfill"] = {
        "report_json": metadata_report["outputs"]["json"],
        "report_csv": metadata_report["outputs"]["csv"],
        "rows_backfilled": metadata_report["totals"]["rows_backfilled"],
        "episodes_backfilled": metadata_report["totals"]["episodes_backfilled"],
        "remaining_unknown_episodes": metadata_report["totals"]["remaining_unknown_episodes"],
        "conflict_episodes": metadata_report["totals"]["conflict_episodes"],
        "metadata_source_coverage_by_batch": metadata_report["metadata_source_coverage_by_batch"],
    }
    manifest["utility_counts"] = {}
    manifest["security_counts"] = {}
    for split_name, episodes in manifest.get("episode_ids", {}).items():
        split_df = dataset[dataset["episode_id"].astype(str).isin([str(value) for value in episodes])]
        manifest["utility_counts"][split_name] = _counts_for_split(split_df, "utility")
        manifest["security_counts"][split_name] = _counts_for_split(split_df, "security")
    manifest["episode_overlap_check"] = _split_overlap_check(manifest)
    manifest["row_counts"] = {
        split: int(len(dataset[dataset["episode_id"].astype(str).isin([str(value) for value in episodes])]))
        for split, episodes in manifest.get("episode_ids", {}).items()
    }
    manifest["label_counts"] = {
        split: _value_counts(dataset[dataset["episode_id"].astype(str).isin([str(value) for value in episodes])], "future_risk_label")
        for split, episodes in manifest.get("episode_ids", {}).items()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _dataset_path_for_manifest(manifest_path: str | Path, fallback_dataset_path: str | Path) -> Path:
    manifest = _load_json(manifest_path)
    manifest_input = manifest.get("input")
    if manifest_input and Path(manifest_input).exists():
        return Path(manifest_input)
    return Path(fallback_dataset_path)


def _write_episode_report_csv(path: str | Path, episode_rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in episode_rows:
            writer.writerow({
                "episode_id": row["episode_id"],
                "metadata_status": row["metadata_status"],
                "utility": row["utility"],
                "security": row["security"],
                "metadata_source_batch": row["metadata_source_batch"],
                "source_count": row["source_count"],
                "conflict_details": json.dumps(row["conflict_details"], sort_keys=True),
            })


def _budget_limited_tasks(audit_path: str | Path | None) -> set[str]:
    if audit_path is None or not Path(audit_path).exists():
        return set()
    audit = _load_json(audit_path)
    return set(str(value) for value in audit.get("effective_after_recovery", {}).get("budget_limited_tasks", []))


def backfill_agentdojo_metadata(
    dataset_paths: list[str | Path] | None = None,
    combined_manifest: str | Path = f"{DEFAULT_COMBINED_SPLIT_DIR}/split_manifest.json",
    disagreement_manifest: str | Path = f"{DEFAULT_DISAGREEMENT_SPLIT_DIR}/split_manifest.json",
    audit_path: str | Path = DEFAULT_AUDIT,
    json_report: str | Path = DEFAULT_JSON_REPORT,
    csv_report: str | Path = DEFAULT_CSV_REPORT,
    summary_patterns: list[str] | None = None,
) -> dict[str, Any]:
    dataset_paths = [Path(path) for path in (dataset_paths or DATASET_PATHS)]
    existing_dataset_paths = [path for path in dataset_paths if path.exists()]
    if not existing_dataset_paths:
        raise FileNotFoundError("No full AgentDojo v2 dataset CSVs found to backfill.")

    budget_limited = _budget_limited_tasks(audit_path)
    summary_paths = _collect_run_summaries(summary_patterns)
    metadata, source_coverage = _build_metadata_map(summary_paths, budget_limited)
    all_episode_ids: set[str] = set()
    dataset_updates = {}
    total_rows_backfilled = 0
    total_episode_backfilled: set[str] = set()

    original_assignments = {}
    for manifest_path in [combined_manifest, disagreement_manifest]:
        path = Path(manifest_path)
        if path.exists():
            original_assignments[str(path)] = _load_json(path).get("episode_ids", {})

    for path in existing_dataset_paths:
        original = pd.read_csv(path)
        original_rows = int(len(original))
        all_episode_ids.update(str(value) for value in original["episode_id"].dropna().unique())
        updated, stats = _apply_metadata(original, metadata)
        if int(len(updated)) != original_rows:
            raise ValueError(f"Row count changed while backfilling {path}.")
        updated.to_csv(path, index=False)
        dataset_updates[str(path)] = {
            "rows": original_rows,
            **stats,
        }
        total_rows_backfilled += stats["rows_backfilled"]
        if stats["episodes_backfilled"]:
            before = original["utility"].map(_normalize_bool) if "utility" in original.columns else pd.Series([None] * len(original))
            after = updated["utility"].map(_normalize_bool)
            mask = before.isna() & after.notna()
            total_episode_backfilled.update(updated.loc[mask, "episode_id"].astype(str).tolist())

    episode_rows = []
    for episode_id in sorted(all_episode_ids):
        item = metadata.get(episode_id)
        if item is None:
            item = {
                "metadata_status": "unknown_no_summary",
                "utility": None,
                "security": None,
                "metadata_source_batch": "",
                "sources": [],
                "conflict_details": {},
            }
        episode_rows.append({
            "episode_id": episode_id,
            "metadata_status": item["metadata_status"],
            "utility": item["utility"],
            "security": item["security"],
            "metadata_source_batch": item["metadata_source_batch"],
            "source_count": len(item["sources"]),
            "conflict_details": item["conflict_details"],
        })

    status_counts = Counter(row["metadata_status"] for row in episode_rows)
    remaining_unknown = sum(
        1 for row in episode_rows if row["metadata_status"] in {"unknown_no_summary", "unknown_not_evaluated"}
    )
    report = {
        "status": "ok",
        "will_call_provider": False,
        "will_run_agentdojo": False,
        "outputs": {
            "json": str(json_report),
            "csv": str(csv_report),
        },
        "inputs": {
            "datasets": [str(path) for path in existing_dataset_paths],
            "summary_paths": [str(path) for path in summary_paths],
            "audit": str(audit_path),
        },
        "metadata_source_coverage_by_batch": source_coverage,
        "episode_status_counts": dict(sorted(status_counts.items())),
        "totals": {
            "episode_count": len(episode_rows),
            "rows_backfilled": total_rows_backfilled,
            "episodes_backfilled": len(total_episode_backfilled),
            "remaining_unknown_episodes": remaining_unknown,
            "conflict_episodes": int(status_counts.get("conflict", 0)),
            "budget_limited_tasks": sorted(budget_limited),
        },
        "datasets": dataset_updates,
        "episodes": episode_rows,
        "validation": {
            "train_val_test_assignment_unchanged": True,
            "row_counts_unchanged": True,
        },
    }
    json_report = Path(json_report)
    json_report.parent.mkdir(parents=True, exist_ok=True)
    json_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_episode_report_csv(csv_report, episode_rows)

    updated_manifests = {}
    for manifest_path, fallback_dataset_path in [
        (combined_manifest, DEFAULT_FULL_COMBINED),
        (disagreement_manifest, DEFAULT_FULL_DISAGREEMENT),
    ]:
        if Path(manifest_path).exists():
            dataset_path = _dataset_path_for_manifest(manifest_path, fallback_dataset_path)
            if not dataset_path.exists():
                continue
            updated = _update_manifest(manifest_path, dataset_path, report)
            updated_manifests[str(manifest_path)] = updated.get("metadata_backfill", {})
            if original_assignments.get(str(Path(manifest_path))) != updated.get("episode_ids", {}):
                report["validation"]["train_val_test_assignment_unchanged"] = False

    json_report.write_text(json.dumps({**report, "updated_manifests": updated_manifests}, indent=2, sort_keys=True), encoding="utf-8")
    return {**report, "updated_manifests": updated_manifests}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill utility/security metadata into full AgentDojo v2 datasets.")
    parser.add_argument("--dataset", action="append", dest="dataset_paths", default=None)
    parser.add_argument("--combined-manifest", default=f"{DEFAULT_COMBINED_SPLIT_DIR}/split_manifest.json")
    parser.add_argument("--disagreement-manifest", default=f"{DEFAULT_DISAGREEMENT_SPLIT_DIR}/split_manifest.json")
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    parser.add_argument("--json-report", default=DEFAULT_JSON_REPORT)
    parser.add_argument("--csv-report", default=DEFAULT_CSV_REPORT)
    args = parser.parse_args()

    result = backfill_agentdojo_metadata(
        dataset_paths=args.dataset_paths,
        combined_manifest=args.combined_manifest,
        disagreement_manifest=args.disagreement_manifest,
        audit_path=args.audit,
        json_report=args.json_report,
        csv_report=args.csv_report,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
