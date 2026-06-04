from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


WITHOUT_DIR = Path("outputs/agentdojo_full_combined")
WITH_DIR = Path("outputs/agentdojo_full_disagreement")
DEFAULT_UTILITY_SLICE_JSON = Path("outputs/agentdojo_full_utility_slice_eval.json")
DEFAULT_CSV_OUT = Path("outputs/agentdojo_full_validation_ablation_summary.csv")
DEFAULT_JSON_OUT = Path("outputs/agentdojo_full_validation_ablation_summary.json")
LIMITATION = "Validation-only full v2 ablation summary, not a final benchmark result."
SCORE_COLUMN_TO_MODEL = {
    "risk_score_logistic": "logistic",
    "risk_score_random_forest": "random_forest",
    "risk_score_hist_gradient_boosting": "hist_gradient_boosting",
}
SLICE_PREFIXES = {
    "all": "utility_all",
    "utility=true": "utility_true",
    "utility=false": "utility_false",
    "unknown_no_summary": "unknown_no_summary",
}
SUMMARY_COLUMNS = [
    "setting",
    "selected_score_column",
    "selected_calibration_method",
    "AUROC",
    "AUPRC",
    "F1@50",
    "Brier score",
    "ECE",
    "pre_risk_auroc",
    "pre_risk_auprc",
    "strict_positive_lead_time_rate",
    "opportunity_adjusted_positive_lead_time_rate",
    "operational_contained_rate",
    "missed_with_pre_risk_window_count",
    "false_alert_episode_rate",
    "recommended_verify_threshold",
    "recommended_alert_threshold",
    "recommended_block_threshold",
    "utility_all_AUROC",
    "utility_all_AUPRC",
    "utility_all_pre_risk_auroc",
    "utility_all_pre_risk_auprc",
    "utility_true_AUROC",
    "utility_true_AUPRC",
    "utility_true_pre_risk_auroc",
    "utility_true_pre_risk_auprc",
    "utility_false_AUROC",
    "utility_false_AUPRC",
    "utility_false_pre_risk_auroc",
    "utility_false_pre_risk_auprc",
    "utility_false_episode_count",
    "utility_false_safe_episode_count",
    "utility_false_warning",
    "unknown_no_summary_AUROC",
    "unknown_no_summary_AUPRC",
    "unknown_no_summary_pre_risk_auroc",
    "unknown_no_summary_pre_risk_auprc",
    "test_split_used",
    "limitation",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_metric(metrics: dict[str, Any], score_column: str, key: str) -> float | None:
    model_name = SCORE_COLUMN_TO_MODEL.get(score_column)
    if model_name is None:
        raise ValueError(f"Unsupported selected score column: {score_column}")
    model_metrics = metrics.get(model_name)
    if not isinstance(model_metrics, dict):
        raise ValueError(f"Missing metrics for selected model: {model_name}")
    return model_metrics.get(key)


def _calibration_metric(calibration: dict[str, Any], prefix: str, method: str) -> float | None:
    return calibration.get(f"{prefix}_{method}")


def _is_test_split_used(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(_is_test_split_used(value) for value in payload.values()) or bool(payload.get("test_split_used", False))
    if isinstance(payload, list):
        return any(_is_test_split_used(item) for item in payload)
    return False


def _payload_warnings(setting: str, name: str, payload: dict[str, Any]) -> list[str]:
    return [f"{setting}:{name}: {warning}" for warning in payload.get("warnings", [])]


def _slice_lookup(utility_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = utility_payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("Utility slice payload must contain a list under 'rows'.")
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        setting = row.get("setting")
        slice_name = row.get("slice")
        if isinstance(setting, str) and isinstance(slice_name, str):
            lookup[(setting, slice_name)] = row
    return lookup


def _add_slice_columns(
    row: dict[str, Any],
    setting: str,
    lookup: dict[tuple[str, str], dict[str, Any]],
    warnings: list[str],
) -> None:
    for slice_name, prefix in SLICE_PREFIXES.items():
        slice_row = lookup.get((setting, slice_name))
        if slice_row is None:
            warnings.append(f"{setting}:{slice_name}: missing utility slice diagnostics.")
            continue
        row[f"{prefix}_AUROC"] = slice_row.get("auroc")
        row[f"{prefix}_AUPRC"] = slice_row.get("auprc")
        row[f"{prefix}_pre_risk_auroc"] = slice_row.get("pre_risk_auroc")
        row[f"{prefix}_pre_risk_auprc"] = slice_row.get("pre_risk_auprc")
        if slice_name == "utility=false":
            safe_episode_count = slice_row.get("safe_episode_count")
            row["utility_false_episode_count"] = slice_row.get("episode_count")
            row["utility_false_safe_episode_count"] = safe_episode_count
            if safe_episode_count == 0:
                warning = (
                    f"{setting}:utility=false has no safe episodes; false-alert and pre-risk "
                    "slice diagnostics are limited."
                )
                row["utility_false_warning"] = warning
                warnings.append(warning)
            else:
                row["utility_false_warning"] = None


def _setting_row(
    setting: str,
    root: Path,
    utility_lookup: dict[tuple[str, str], dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    risk_metrics = _load_json(root / "agentdojo_split_risk_estimator_metrics.json")
    calibration = _load_json(root / "agentdojo_calibration_metrics_val.json")
    early_warning = _load_json(root / "agentdojo_early_warning_metrics_val.json")
    threshold_sweep = _load_json(root / "agentdojo_threshold_sweep_val.json")
    recommended = threshold_sweep.get("recommended_policy") or {}

    for name, payload in [
        ("risk_metrics", risk_metrics),
        ("calibration", calibration),
        ("early_warning", early_warning),
        ("threshold_sweep", threshold_sweep),
    ]:
        warnings.extend(_payload_warnings(setting, name, payload))

    score_column = calibration.get("score_column")
    calibration_method = calibration.get("selected_calibration_method")
    if not isinstance(score_column, str):
        raise ValueError(f"{setting}: missing selected score column.")
    if not isinstance(calibration_method, str):
        raise ValueError(f"{setting}: missing selected calibration method.")

    row: dict[str, Any] = {
        "setting": setting,
        "selected_score_column": score_column,
        "selected_calibration_method": calibration_method,
        "AUROC": _selected_metric(risk_metrics, score_column, "auroc"),
        "AUPRC": _selected_metric(risk_metrics, score_column, "auprc"),
        "F1@50": _selected_metric(risk_metrics, score_column, "f1_at_50"),
        "Brier score": _calibration_metric(calibration, "brier_score", calibration_method),
        "ECE": _calibration_metric(calibration, "ece", calibration_method),
        "pre_risk_auroc": early_warning.get("pre_risk_auroc"),
        "pre_risk_auprc": early_warning.get("pre_risk_auprc"),
        "strict_positive_lead_time_rate": recommended.get("strict_positive_lead_time_rate"),
        "opportunity_adjusted_positive_lead_time_rate": recommended.get(
            "opportunity_adjusted_positive_lead_time_rate"
        ),
        "operational_contained_rate": recommended.get("operational_contained_rate"),
        "missed_with_pre_risk_window_count": recommended.get("missed_with_pre_risk_window_count"),
        "false_alert_episode_rate": recommended.get("false_alert_episode_rate"),
        "recommended_verify_threshold": recommended.get("verify_threshold"),
        "recommended_alert_threshold": recommended.get("alert_threshold"),
        "recommended_block_threshold": recommended.get("block_threshold"),
        "test_split_used": any(
            _is_test_split_used(payload)
            for payload in [risk_metrics, calibration, early_warning, threshold_sweep]
        ),
        "limitation": LIMITATION,
    }
    _add_slice_columns(row, setting, utility_lookup, warnings)
    return row


def _numeric_deltas(without_row: dict[str, Any], with_row: dict[str, Any]) -> dict[str, float | int]:
    deltas: dict[str, float | int] = {}
    for column in SUMMARY_COLUMNS:
        left = without_row.get(column)
        right = with_row.get(column)
        if isinstance(left, bool) or isinstance(right, bool):
            continue
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            deltas[column] = round(right - left, 12)
    return deltas


def summarize_agentdojo_full_validation_ablation(
    without_dir: str | Path = WITHOUT_DIR,
    with_dir: str | Path = WITH_DIR,
    utility_slice_json: str | Path = DEFAULT_UTILITY_SLICE_JSON,
    csv_out: str | Path = DEFAULT_CSV_OUT,
    json_out: str | Path = DEFAULT_JSON_OUT,
) -> dict[str, Any]:
    without_dir = Path(without_dir)
    with_dir = Path(with_dir)
    utility_slice_json = Path(utility_slice_json)
    csv_out = Path(csv_out)
    json_out = Path(json_out)

    utility_payload = _load_json(utility_slice_json)
    warnings: list[str] = list(utility_payload.get("warnings", []))
    utility_lookup = _slice_lookup(utility_payload)
    rows = [
        _setting_row("without_disagreement", without_dir, utility_lookup, warnings),
        _setting_row("with_disagreement", with_dir, utility_lookup, warnings),
    ]
    test_split_used = any(bool(row["test_split_used"]) for row in rows) or _is_test_split_used(utility_payload)
    if test_split_used:
        warnings.append("At least one input reported test_split_used=true; output is not validation-only clean.")

    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)

    result = {
        "status": "ok",
        "limitation": LIMITATION,
        "validation_only": True,
        "test_split_used": test_split_used,
        "outputs": {
            "csv": str(csv_out),
            "json": str(json_out),
        },
        "inputs": {
            "without_disagreement": str(without_dir),
            "with_disagreement": str(with_dir),
            "utility_slice_json": str(utility_slice_json),
        },
        "rows": rows,
        "deltas_with_minus_without": _numeric_deltas(rows[0], rows[1]),
        "warnings": warnings,
        "rules": {
            "validation_only": True,
            "will_call_provider": False,
            "will_fit_calibration": False,
            "will_run_agentdojo": False,
            "will_train_risk_estimator": False,
        },
    }
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize full v2 validation-only AgentDojo ablation.")
    parser.add_argument("--without-dir", default=str(WITHOUT_DIR))
    parser.add_argument("--with-dir", default=str(WITH_DIR))
    parser.add_argument("--utility-slice-json", default=str(DEFAULT_UTILITY_SLICE_JSON))
    parser.add_argument("--csv-out", default=str(DEFAULT_CSV_OUT))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    args = parser.parse_args()

    result = summarize_agentdojo_full_validation_ablation(
        without_dir=args.without_dir,
        with_dir=args.with_dir,
        utility_slice_json=args.utility_slice_json,
        csv_out=args.csv_out,
        json_out=args.json_out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
