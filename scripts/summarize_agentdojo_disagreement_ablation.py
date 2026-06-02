from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


WITHOUT_DIR = Path("outputs/agentdojo_multisuite_combined")
WITH_DIR = Path("outputs/agentdojo_multisuite_disagreement")
DEFAULT_CSV_OUT = "outputs/agentdojo_validation_disagreement_ablation_summary.csv"
DEFAULT_JSON_OUT = "outputs/agentdojo_validation_disagreement_ablation_summary.json"
LIMITATION = "Validation-only disagreement ablation sanity check, not a final benchmark result."
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
    "positive_lead_time_contained_count",
    "strict_positive_lead_time_rate",
    "no_window_just_in_time_contained_count",
    "operational_contained_count",
    "operational_contained_rate",
    "opportunity_adjusted_positive_lead_time_rate",
    "missed_with_pre_risk_window_count",
    "false_alert_episode_count",
    "false_alert_episode_rate",
    "recommended_verify_threshold",
    "recommended_alert_threshold",
    "recommended_block_threshold",
    "test_split_used",
    "limitation",
]
SCORE_COLUMN_TO_MODEL = {
    "risk_score_logistic": "logistic",
    "risk_score_random_forest": "random_forest",
    "risk_score_hist_gradient_boosting": "hist_gradient_boosting",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_metric(metrics: dict[str, Any], score_column: str, key: str) -> float | None:
    model_name = SCORE_COLUMN_TO_MODEL.get(score_column)
    if model_name is None:
        raise ValueError(f"Unsupported selected score column: {score_column}")
    return metrics.get(model_name, {}).get(key)


def _calibration_metric(calibration: dict[str, Any], prefix: str, method: str) -> float | None:
    return calibration.get(f"{prefix}_{method}")


def _setting_row(setting: str, root: Path) -> dict[str, Any]:
    risk_metrics = _load_json(root / "agentdojo_split_risk_estimator_metrics.json")
    calibration = _load_json(root / "agentdojo_calibration_metrics_val.json")
    early_warning = _load_json(root / "agentdojo_early_warning_metrics_val.json")
    threshold_sweep = _load_json(root / "agentdojo_threshold_sweep_val.json")
    recommended = threshold_sweep.get("recommended_policy") or {}

    score_column = calibration.get("score_column")
    calibration_method = calibration.get("selected_calibration_method")
    test_split_used = any(
        bool(payload.get("test_split_used", False))
        for payload in [risk_metrics, calibration, early_warning, threshold_sweep, recommended]
    )
    return {
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
        "positive_lead_time_contained_count": recommended.get("positive_lead_time_contained_count"),
        "strict_positive_lead_time_rate": recommended.get("strict_positive_lead_time_rate"),
        "no_window_just_in_time_contained_count": recommended.get("no_window_just_in_time_contained_count"),
        "operational_contained_count": recommended.get("operational_contained_count"),
        "operational_contained_rate": recommended.get("operational_contained_rate"),
        "opportunity_adjusted_positive_lead_time_rate": recommended.get("opportunity_adjusted_positive_lead_time_rate"),
        "missed_with_pre_risk_window_count": recommended.get("missed_with_pre_risk_window_count"),
        "false_alert_episode_count": recommended.get("false_alert_episode_count"),
        "false_alert_episode_rate": recommended.get("false_alert_episode_rate"),
        "recommended_verify_threshold": recommended.get("verify_threshold"),
        "recommended_alert_threshold": recommended.get("alert_threshold"),
        "recommended_block_threshold": recommended.get("block_threshold"),
        "test_split_used": test_split_used,
        "limitation": LIMITATION,
    }


def _numeric_deltas(without_row: dict[str, Any], with_row: dict[str, Any]) -> dict[str, float | int | None]:
    deltas: dict[str, float | int | None] = {}
    for column in SUMMARY_COLUMNS:
        left = without_row.get(column)
        right = with_row.get(column)
        if isinstance(left, bool) or isinstance(right, bool):
            continue
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            deltas[column] = right - left
    return deltas


def summarize_agentdojo_disagreement_ablation(
    without_dir: str | Path = WITHOUT_DIR,
    with_dir: str | Path = WITH_DIR,
    csv_out: str | Path = DEFAULT_CSV_OUT,
    json_out: str | Path = DEFAULT_JSON_OUT,
) -> dict[str, Any]:
    without_dir = Path(without_dir)
    with_dir = Path(with_dir)
    rows = [
        _setting_row("without_disagreement", without_dir),
        _setting_row("with_disagreement", with_dir),
    ]
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    csv_out = Path(csv_out)
    json_out = Path(json_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    result = {
        "status": "ok",
        "limitation": LIMITATION,
        "outputs": {
            "csv": str(csv_out),
            "json": str(json_out),
        },
        "inputs": {
            "without_disagreement": str(without_dir),
            "with_disagreement": str(with_dir),
        },
        "rows": rows,
        "deltas_with_minus_without": _numeric_deltas(rows[0], rows[1]),
        "test_split_used": any(bool(row["test_split_used"]) for row in rows),
    }
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize validation-only AgentDojo disagreement ablation.")
    parser.add_argument("--without-dir", default=str(WITHOUT_DIR))
    parser.add_argument("--with-dir", default=str(WITH_DIR))
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    args = parser.parse_args()

    result = summarize_agentdojo_disagreement_ablation(
        without_dir=args.without_dir,
        with_dir=args.with_dir,
        csv_out=args.csv_out,
        json_out=args.json_out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
