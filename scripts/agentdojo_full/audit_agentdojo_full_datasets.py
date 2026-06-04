from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.thermometer_baseline import select_feature_columns  # noqa: E402


DEFAULT_COMBINED = "outputs/agentdojo_full_combined/agentdojo_full_prefix_dataset.csv"
DEFAULT_DISAGREEMENT = "outputs/agentdojo_full_disagreement/agentdojo_full_disagreement_prefix_dataset.csv"
DEFAULT_COMBINED_MANIFEST = "outputs/agentdojo_full_combined/splits/split_manifest.json"
DEFAULT_DISAGREEMENT_MANIFEST = "outputs/agentdojo_full_disagreement/splits/split_manifest.json"
DEFAULT_BACKFILL_REPORT = "outputs/agentdojo_metadata_backfill_report.json"
DEFAULT_JSON_OUT = "outputs/agentdojo_full_dataset_audit.json"
DEFAULT_CSV_OUT = "outputs/agentdojo_full_dataset_audit.csv"
SPLITS = ["train", "val", "test"]
SEMANTIC_KEY = ["episode_id", "step_id", "hook_type"]
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
LEAKAGE_COLUMNS = [
    "future_risk_label",
    "future_severity",
    "t_risk",
    "lead_time_if_alert_now",
]
DIAGNOSTIC_ONLY_COLUMNS = [
    "oracle_violation",
    "utility",
    "security",
    "metadata_status",
    "metadata_source_batch",
]
CSV_COLUMNS = [
    "dataset",
    "split",
    "rows",
    "episodes",
    "future_risk_label_counts",
    "source_suite_counts",
    "source_batch_counts",
    "utility_counts",
    "security_counts",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = Counter("" if pd.isna(value) else str(value) for value in df[column].tolist())
    return dict(sorted(counts.items()))


def _metadata_bucket(row: pd.Series, column: str) -> str:
    value = row.get(column)
    if value is True or str(value).strip().lower() == "true":
        return "true"
    if value is False or str(value).strip().lower() == "false":
        return "false"
    status = str(row.get("metadata_status", "")).strip()
    if status in {"unknown_no_summary", "unknown_not_evaluated", "conflict"}:
        return status
    return "unknown_no_summary"


def _metadata_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    counts = Counter(_metadata_bucket(row, column) for _, row in df.iterrows())
    return {
        key: int(counts.get(key, 0))
        for key in ["true", "false", "unknown_no_summary", "unknown_not_evaluated", "conflict"]
    }


def _subset_distribution(df: pd.DataFrame, subset_column: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for bucket in ["true", "false", "unknown_no_summary", "unknown_not_evaluated", "conflict"]:
        mask = df.apply(lambda row: _metadata_bucket(row, subset_column) == bucket, axis=1)
        subset = df[mask]
        result[bucket] = {
            "rows": int(len(subset)),
            "episodes": int(subset["episode_id"].nunique()) if "episode_id" in subset.columns else 0,
            "future_risk_label_counts": _value_counts(subset, "future_risk_label"),
            "source_suite_counts": _value_counts(subset, "source_suite"),
        }
    return result


def _episode_overlap(episode_ids: dict[str, list[str]]) -> dict[str, Any]:
    sets = {split: set(str(value) for value in episode_ids.get(split, [])) for split in SPLITS}
    overlaps = {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }
    return {"passed": not any(overlaps.values()), "overlaps": overlaps}


def _split_df(df: pd.DataFrame, manifest: dict[str, Any], split: str) -> pd.DataFrame:
    episodes = [str(value) for value in manifest["episode_ids"].get(split, [])]
    return df[df["episode_id"].astype(str).isin(episodes)].copy()


def _split_summary(df: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for split in SPLITS:
        split_df = _split_df(df, manifest, split)
        result[split] = {
            "rows": int(len(split_df)),
            "episodes": int(split_df["episode_id"].nunique()),
            "future_risk_label_counts": _value_counts(split_df, "future_risk_label"),
            "source_suite_counts": _value_counts(split_df, "source_suite"),
            "source_batch_counts": _value_counts(split_df, "source_batch"),
            "utility_counts": _metadata_counts(split_df, "utility"),
            "security_counts": _metadata_counts(split_df, "security"),
        }
    return result


def _leakage_audit(df: pd.DataFrame) -> dict[str, Any]:
    features = select_feature_columns(df)
    diagnostic_present = [column for column in DIAGNOSTIC_ONLY_COLUMNS if column in features]
    leakage_present = [column for column in LEAKAGE_COLUMNS if column in features]
    return {
        "selected_feature_columns": features,
        "future_risk_label_not_used_as_feature": "future_risk_label" not in features,
        "future_severity_not_used_as_feature": "future_severity" not in features,
        "t_risk_not_used_as_feature": "t_risk" not in features,
        "lead_time_if_alert_now_not_used_as_feature": "lead_time_if_alert_now" not in features,
        "oracle_violation_diagnostic_only": "oracle_violation" not in features,
        "utility_security_diagnostic_only": "utility" not in features and "security" not in features,
        "metadata_diagnostic_only": "metadata_status" not in features and "metadata_source_batch" not in features,
        "leakage_columns_present_in_features": leakage_present,
        "diagnostic_columns_present_in_features": diagnostic_present,
        "passed": not leakage_present and not diagnostic_present,
    }


def _write_csv(path: str | Path, combined: dict[str, Any], disagreement: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for dataset_name, summary in [("no_disagreement", combined), ("disagreement", disagreement)]:
            for split, row in summary.items():
                writer.writerow({
                    "dataset": dataset_name,
                    "split": split,
                    "rows": row["rows"],
                    "episodes": row["episodes"],
                    "future_risk_label_counts": json.dumps(row["future_risk_label_counts"], sort_keys=True),
                    "source_suite_counts": json.dumps(row["source_suite_counts"], sort_keys=True),
                    "source_batch_counts": json.dumps(row["source_batch_counts"], sort_keys=True),
                    "utility_counts": json.dumps(row["utility_counts"], sort_keys=True),
                    "security_counts": json.dumps(row["security_counts"], sort_keys=True),
                })


def audit_agentdojo_full_datasets(
    combined_path: str | Path = DEFAULT_COMBINED,
    disagreement_path: str | Path = DEFAULT_DISAGREEMENT,
    combined_manifest_path: str | Path = DEFAULT_COMBINED_MANIFEST,
    disagreement_manifest_path: str | Path = DEFAULT_DISAGREEMENT_MANIFEST,
    backfill_report_path: str | Path = DEFAULT_BACKFILL_REPORT,
    json_out: str | Path = DEFAULT_JSON_OUT,
    csv_out: str | Path = DEFAULT_CSV_OUT,
) -> dict[str, Any]:
    combined_df = pd.read_csv(combined_path)
    disagreement_df = pd.read_csv(disagreement_path)
    combined_manifest = _load_json(combined_manifest_path)
    disagreement_manifest = _load_json(disagreement_manifest_path)
    backfill_report = _load_json(backfill_report_path) if Path(backfill_report_path).exists() else {}

    combined_episode_set = set(combined_df["episode_id"].astype(str).unique())
    disagreement_episode_set = set(disagreement_df["episode_id"].astype(str).unique())
    combined_summary = _split_summary(combined_df, combined_manifest)
    disagreement_summary = _split_summary(disagreement_df, disagreement_manifest)
    warnings: list[str] = []
    if backfill_report.get("totals", {}).get("remaining_unknown_episodes", 0):
        warnings.append("Remaining unknown_no_summary episodes are tracked as a diagnostic group.")
    if set(DISAGREEMENT_COLUMNS) & set(combined_df.columns):
        warnings.append("No-disagreement dataset contains disagreement columns.")
    missing_disagreement_columns = [column for column in DISAGREEMENT_COLUMNS if column not in disagreement_df.columns]
    if missing_disagreement_columns:
        warnings.append("Disagreement dataset is missing required disagreement columns.")

    payload = {
        "status": "ok",
        "inputs": {
            "combined": str(combined_path),
            "disagreement": str(disagreement_path),
            "combined_manifest": str(combined_manifest_path),
            "disagreement_manifest": str(disagreement_manifest_path),
            "backfill_report": str(backfill_report_path),
        },
        "outputs": {
            "json": str(json_out),
            "csv": str(csv_out),
        },
        "episode_sets_match": combined_episode_set == disagreement_episode_set,
        "split_assignment_matches": combined_manifest.get("episode_ids") == disagreement_manifest.get("episode_ids"),
        "episode_overlap_check": _episode_overlap(combined_manifest.get("episode_ids", {})),
        "row_counts": {
            "no_disagreement": int(len(combined_df)),
            "disagreement": int(len(disagreement_df)),
        },
        "episode_counts": {
            "no_disagreement": int(combined_df["episode_id"].nunique()),
            "disagreement": int(disagreement_df["episode_id"].nunique()),
        },
        "semantic_duplicate_rows": {
            "no_disagreement": int(combined_df.duplicated(subset=SEMANTIC_KEY, keep=False).sum()),
            "disagreement": int(disagreement_df.duplicated(subset=SEMANTIC_KEY, keep=False).sum()),
        },
        "splits": {
            "no_disagreement": combined_summary,
            "disagreement": disagreement_summary,
        },
        "utility_subset_distributions": {
            "no_disagreement": _subset_distribution(combined_df, "utility"),
            "disagreement": _subset_distribution(disagreement_df, "utility"),
        },
        "security_subset_distributions": {
            "no_disagreement": _subset_distribution(combined_df, "security"),
            "disagreement": _subset_distribution(disagreement_df, "security"),
        },
        "column_checks": {
            "required_disagreement_columns_exist": not missing_disagreement_columns,
            "missing_disagreement_columns": missing_disagreement_columns,
            "no_disagreement_has_disagreement_columns": bool(set(DISAGREEMENT_COLUMNS) & set(combined_df.columns)),
        },
        "leakage_audit": {
            "no_disagreement": _leakage_audit(combined_df),
            "disagreement": _leakage_audit(disagreement_df),
            "notes": {
                "oracle_violation": "Diagnostic only; not a model feature.",
                "utility_security": "Diagnostic subset columns only; not model features.",
                "test_split": "Do not use test split for model selection.",
            },
        },
        "metadata_backfill": backfill_report.get("totals", {}),
        "training_readiness": {
            "ready": True,
            "unknown_no_summary_blocks_training": False,
            "unknown_no_summary_note": (
                "Remaining utility/security unknown_no_summary rows are diagnostic grouping gaps, "
                "not missing trace/prefix labels."
            ),
        },
        "warnings": warnings,
        "rules": {
            "will_call_provider": False,
            "will_run_agentdojo": False,
            "will_train_risk_estimator": False,
            "will_fit_calibration": False,
            "test_split_used_for_model_selection": False,
        },
    }
    readiness_checks = [
        payload["episode_sets_match"],
        payload["split_assignment_matches"],
        payload["episode_overlap_check"]["passed"],
        payload["semantic_duplicate_rows"]["no_disagreement"] == 0,
        payload["semantic_duplicate_rows"]["disagreement"] == 0,
        payload["column_checks"]["required_disagreement_columns_exist"],
        not payload["column_checks"]["no_disagreement_has_disagreement_columns"],
        payload["leakage_audit"]["no_disagreement"]["passed"],
        payload["leakage_audit"]["disagreement"]["passed"],
    ]
    payload["training_readiness"]["ready"] = all(readiness_checks)

    json_out = Path(json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_out, combined_summary, disagreement_summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit full AgentDojo v2 datasets before training.")
    parser.add_argument("--combined", default=DEFAULT_COMBINED)
    parser.add_argument("--disagreement", default=DEFAULT_DISAGREEMENT)
    parser.add_argument("--combined-manifest", default=DEFAULT_COMBINED_MANIFEST)
    parser.add_argument("--disagreement-manifest", default=DEFAULT_DISAGREEMENT_MANIFEST)
    parser.add_argument("--backfill-report", default=DEFAULT_BACKFILL_REPORT)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    args = parser.parse_args()

    result = audit_agentdojo_full_datasets(
        combined_path=args.combined,
        disagreement_path=args.disagreement,
        combined_manifest_path=args.combined_manifest,
        disagreement_manifest_path=args.disagreement_manifest,
        backfill_report_path=args.backfill_report,
        json_out=args.json_out,
        csv_out=args.csv_out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
