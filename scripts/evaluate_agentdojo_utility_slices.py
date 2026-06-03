from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_agentdojo_early_warning_metrics import evaluate_early_warning  # noqa: E402


DEFAULT_WITHOUT_SCORES = "outputs/agentdojo_full_combined/agentdojo_thermometer_scores_val.csv"
DEFAULT_WITHOUT_VAL = "outputs/agentdojo_full_combined/splits/agentdojo_val.csv"
DEFAULT_WITHOUT_SWEEP = "outputs/agentdojo_full_combined/agentdojo_threshold_sweep_val.json"
DEFAULT_WITH_SCORES = "outputs/agentdojo_full_disagreement/agentdojo_thermometer_scores_val.csv"
DEFAULT_WITH_VAL = "outputs/agentdojo_full_disagreement/splits/agentdojo_val.csv"
DEFAULT_WITH_SWEEP = "outputs/agentdojo_full_disagreement/agentdojo_threshold_sweep_val.json"
DEFAULT_CSV_OUT = "outputs/agentdojo_full_utility_slice_eval.csv"
DEFAULT_JSON_OUT = "outputs/agentdojo_full_utility_slice_eval.json"
TARGET = "future_risk_label"
SLICES = ["all", "utility=true", "utility=false", "unknown_no_summary"]
LIMITATION = "Validation-only utility-slice evaluation, not a final benchmark result."


def _metadata_bucket(row: pd.Series) -> str:
    utility = row.get("utility")
    if utility is True or str(utility).strip().lower() == "true":
        return "utility=true"
    if utility is False or str(utility).strip().lower() == "false":
        return "utility=false"
    status = str(row.get("metadata_status", "")).strip()
    if status == "unknown_no_summary":
        return "unknown_no_summary"
    if status == "unknown_not_evaluated":
        return "unknown_not_evaluated"
    if status == "conflict":
        return "conflict"
    return "unknown_no_summary"


def _policy_from_score(score: float, verify_threshold: float, alert_threshold: float, block_threshold: float) -> str:
    if score >= block_threshold:
        return "block"
    if score >= alert_threshold:
        return "alert"
    if score >= verify_threshold:
        return "verify"
    return "watch"


def _safe_metric(fn, labels: pd.Series, scores: pd.Series) -> float | None:
    valid = pd.DataFrame({"label": labels, "score": scores}).dropna()
    if len(valid) == 0 or valid["label"].nunique() < 2:
        return None
    return float(fn(valid["label"].astype(int), valid["score"].astype(float)))


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].fillna("missing").astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _merge_scores_and_val(scores_path: str | Path, val_path: str | Path, sweep_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    scores = pd.read_csv(scores_path)
    val = pd.read_csv(val_path)
    sweep = json.loads(Path(sweep_path).read_text(encoding="utf-8"))
    keys = ["episode_id", "step_id", "hook_type"]
    metadata_columns = [
        column
        for column in [
            *keys,
            "utility",
            "security",
            "metadata_status",
            "metadata_source_batch",
            "source_suite",
            "source_batch",
            "t_risk",
            "lead_time_if_alert_now",
            "future_severity",
        ]
        if column in val.columns
    ]
    merged = scores.merge(val[metadata_columns].drop_duplicates(keys), on=keys, how="left")
    policy = sweep.get("recommended_policy", {})
    verify_threshold = float(policy.get("verify_threshold", 0.0))
    alert_threshold = float(policy.get("alert_threshold", verify_threshold))
    block_threshold = float(policy.get("block_threshold", alert_threshold))
    merged["policy"] = merged["thermometer_score"].astype(float).apply(
        lambda score: _policy_from_score(score, verify_threshold, alert_threshold, block_threshold)
    )
    merged["_utility_slice"] = merged.apply(_metadata_bucket, axis=1)
    return merged, sweep


def _evaluate_slice(setting: str, slice_name: str, df: pd.DataFrame, sweep: dict[str, Any]) -> dict[str, Any]:
    if slice_name == "all":
        subset = df.copy()
    else:
        subset = df[df["_utility_slice"] == slice_name].copy()

    labels = subset[TARGET] if TARGET in subset.columns else pd.Series(dtype=int)
    scores = subset["thermometer_score"] if "thermometer_score" in subset.columns else pd.Series(dtype=float)
    y_pred = (scores.fillna(-1).astype(float) >= float(sweep["recommended_policy"]["alert_threshold"])).astype(int)
    if len(subset) and labels.dropna().nunique() > 0:
        f1_at_alert = float(f1_score(labels.astype(int), y_pred, zero_division=0))
    else:
        f1_at_alert = None
    early_warning, _, ew_warnings = evaluate_early_warning(subset) if len(subset) else ({}, pd.DataFrame(), [])
    score_mean = float(scores.mean()) if len(scores.dropna()) else None

    row = {
        "setting": setting,
        "slice": slice_name,
        "row_count": int(len(subset)),
        "episode_count": int(subset["episode_id"].nunique()) if "episode_id" in subset.columns else 0,
        "future_risk_label_counts": _value_counts(subset, TARGET),
        "source_suite_counts": _value_counts(subset, "source_suite"),
        "source_batch_counts": _value_counts(subset, "source_batch"),
        "utility_slice_counts": _value_counts(subset, "_utility_slice"),
        "thermometer_score_mean": score_mean,
        "auroc": _safe_metric(roc_auc_score, labels, scores),
        "auprc": _safe_metric(average_precision_score, labels, scores),
        "f1_at_recommended_alert": f1_at_alert,
        "recommended_verify_threshold": sweep["recommended_policy"].get("verify_threshold"),
        "recommended_alert_threshold": sweep["recommended_policy"].get("alert_threshold"),
        "recommended_block_threshold": sweep["recommended_policy"].get("block_threshold"),
        "risky_episode_count": early_warning.get("risky_episode_count"),
        "safe_episode_count": early_warning.get("safe_episode_count"),
        "pre_risk_auroc": early_warning.get("pre_risk_auroc"),
        "pre_risk_auprc": early_warning.get("pre_risk_auprc"),
        "operational_contained_count": early_warning.get("operational_contained_count"),
        "operational_contained_rate": early_warning.get("operational_contained_rate"),
        "strict_positive_lead_time_rate": early_warning.get("strict_positive_lead_time_rate"),
        "opportunity_adjusted_positive_lead_time_rate": early_warning.get("opportunity_adjusted_positive_lead_time_rate"),
        "false_alert_episode_count": early_warning.get("false_alert_episode_count"),
        "false_alert_episode_rate": early_warning.get("false_alert_episode_rate"),
        "missed_risk_episode_count": early_warning.get("missed_risk_episode_count"),
        "missed_with_pre_risk_window_count": early_warning.get("missed_with_pre_risk_window_count"),
        "warnings": ew_warnings,
        "test_split_used": False,
        "limitation": LIMITATION,
    }
    return row


def evaluate_agentdojo_utility_slices(
    without_scores: str | Path = DEFAULT_WITHOUT_SCORES,
    without_val: str | Path = DEFAULT_WITHOUT_VAL,
    without_sweep: str | Path = DEFAULT_WITHOUT_SWEEP,
    with_scores: str | Path = DEFAULT_WITH_SCORES,
    with_val: str | Path = DEFAULT_WITH_VAL,
    with_sweep: str | Path = DEFAULT_WITH_SWEEP,
    csv_out: str | Path = DEFAULT_CSV_OUT,
    json_out: str | Path = DEFAULT_JSON_OUT,
) -> dict[str, Any]:
    rows = []
    inputs = {
        "without_disagreement": {
            "scores": str(without_scores),
            "val": str(without_val),
            "threshold_sweep": str(without_sweep),
        },
        "with_disagreement": {
            "scores": str(with_scores),
            "val": str(with_val),
            "threshold_sweep": str(with_sweep),
        },
    }
    for setting, scores_path, val_path, sweep_path in [
        ("without_disagreement", without_scores, without_val, without_sweep),
        ("with_disagreement", with_scores, with_val, with_sweep),
    ]:
        merged, sweep = _merge_scores_and_val(scores_path, val_path, sweep_path)
        for slice_name in SLICES:
            rows.append(_evaluate_slice(setting, slice_name, merged, sweep))

    output_df = pd.DataFrame(rows)
    csv_out = Path(csv_out)
    json_out = Path(json_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(csv_out, index=False)
    payload = {
        "status": "ok",
        "inputs": inputs,
        "outputs": {
            "csv": str(csv_out),
            "json": str(json_out),
        },
        "settings": ["without_disagreement", "with_disagreement"],
        "slices": SLICES,
        "rows": rows,
        "test_split_used": False,
        "limitation": LIMITATION,
        "rules": {
            "validation_only": True,
            "will_call_provider": False,
            "will_run_agentdojo": False,
            "will_train_risk_estimator": False,
            "will_fit_calibration": False,
        },
    }
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full AgentDojo v2 validation utility slices.")
    parser.add_argument("--without-scores", default=DEFAULT_WITHOUT_SCORES)
    parser.add_argument("--without-val", default=DEFAULT_WITHOUT_VAL)
    parser.add_argument("--without-sweep", default=DEFAULT_WITHOUT_SWEEP)
    parser.add_argument("--with-scores", default=DEFAULT_WITH_SCORES)
    parser.add_argument("--with-val", default=DEFAULT_WITH_VAL)
    parser.add_argument("--with-sweep", default=DEFAULT_WITH_SWEEP)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    args = parser.parse_args()

    result = evaluate_agentdojo_utility_slices(
        without_scores=args.without_scores,
        without_val=args.without_val,
        without_sweep=args.without_sweep,
        with_scores=args.with_scores,
        with_val=args.with_val,
        with_sweep=args.with_sweep,
        csv_out=args.csv_out,
        json_out=args.json_out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
