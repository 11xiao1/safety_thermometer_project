import json
from pathlib import Path

import pandas as pd

from scripts.summarize_agentdojo_full_validation_ablation import (
    SUMMARY_COLUMNS,
    summarize_agentdojo_full_validation_ablation,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_inputs(root: Path, *, auroc: float, auprc: float, pre_risk_auroc: float, alert_threshold: float) -> None:
    _write_json(
        root / "agentdojo_split_risk_estimator_metrics.json",
        {
            "hist_gradient_boosting": {
                "auroc": auroc,
                "auprc": auprc,
                "f1_at_50": 0.8,
            },
            "test_split_used": False,
            "warnings": [],
        },
    )
    _write_json(
        root / "agentdojo_calibration_metrics_val.json",
        {
            "score_column": "risk_score_hist_gradient_boosting",
            "selected_calibration_method": "isotonic",
            "brier_score_isotonic": 0.1,
            "ece_isotonic": 0.02,
            "test_split_used": False,
            "warnings": [],
        },
    )
    _write_json(
        root / "agentdojo_early_warning_metrics_val.json",
        {
            "pre_risk_auroc": pre_risk_auroc,
            "pre_risk_auprc": 0.6,
            "test_split_used": False,
            "warnings": [],
        },
    )
    _write_json(
        root / "agentdojo_threshold_sweep_val.json",
        {
            "recommended_policy": {
                "strict_positive_lead_time_rate": 0.7,
                "opportunity_adjusted_positive_lead_time_rate": 0.75,
                "operational_contained_rate": 0.8,
                "missed_with_pre_risk_window_count": 2,
                "false_alert_episode_rate": 0.25,
                "verify_threshold": 35.0,
                "alert_threshold": alert_threshold,
                "block_threshold": 60.0,
                "test_split_used": False,
            },
            "test_split_used": False,
            "warnings": [],
        },
    )


def _slice_row(setting: str, slice_name: str, *, auroc: float, safe_episode_count: int) -> dict:
    return {
        "setting": setting,
        "slice": slice_name,
        "auroc": auroc,
        "auprc": auroc + 0.01,
        "pre_risk_auroc": auroc - 0.1 if safe_episode_count else None,
        "pre_risk_auprc": auroc - 0.05 if safe_episode_count else None,
        "episode_count": 3,
        "safe_episode_count": safe_episode_count,
        "test_split_used": False,
    }


def test_full_validation_ablation_summary_writes_outputs_and_warnings(tmp_path):
    without_dir = tmp_path / "without"
    with_dir = tmp_path / "with"
    _write_inputs(without_dir, auroc=0.8, auprc=0.82, pre_risk_auroc=0.7, alert_threshold=45.0)
    _write_inputs(with_dir, auroc=0.9, auprc=0.92, pre_risk_auroc=0.8, alert_threshold=40.0)
    utility_slice_json = tmp_path / "utility.json"
    _write_json(
        utility_slice_json,
        {
            "rows": [
                _slice_row("without_disagreement", "all", auroc=0.81, safe_episode_count=2),
                _slice_row("without_disagreement", "utility=true", auroc=0.72, safe_episode_count=1),
                _slice_row("without_disagreement", "utility=false", auroc=0.91, safe_episode_count=0),
                _slice_row("without_disagreement", "unknown_no_summary", auroc=0.83, safe_episode_count=1),
                _slice_row("with_disagreement", "all", auroc=0.86, safe_episode_count=2),
                _slice_row("with_disagreement", "utility=true", auroc=0.77, safe_episode_count=1),
                _slice_row("with_disagreement", "utility=false", auroc=0.96, safe_episode_count=0),
                _slice_row("with_disagreement", "unknown_no_summary", auroc=0.88, safe_episode_count=1),
            ],
            "test_split_used": False,
            "warnings": [],
        },
    )

    result = summarize_agentdojo_full_validation_ablation(
        without_dir=without_dir,
        with_dir=with_dir,
        utility_slice_json=utility_slice_json,
        csv_out=tmp_path / "summary.csv",
        json_out=tmp_path / "summary.json",
    )

    summary = pd.read_csv(tmp_path / "summary.csv")
    assert summary.columns.tolist() == SUMMARY_COLUMNS
    assert summary["setting"].tolist() == ["without_disagreement", "with_disagreement"]
    assert result["deltas_with_minus_without"]["AUROC"] == 0.1
    assert result["deltas_with_minus_without"]["pre_risk_auroc"] == 0.1
    assert result["deltas_with_minus_without"]["recommended_alert_threshold"] == -5.0
    assert result["test_split_used"] is False
    assert len(result["warnings"]) == 2
    assert "no safe episodes" in result["warnings"][0]
