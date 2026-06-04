from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labels.judge_schema import example_risky_label  # noqa: E402
from src.labels.label_merger import merge_label_sources  # noqa: E402
from src.labels.prefix_label_propagation import propagated_prefix_label  # noqa: E402


DEFAULT_PREFIX = "outputs/agentdojo_full_combined/splits/agentdojo_val.csv"
DEFAULT_JSON_OUT = "outputs/labels/label_pipeline_dry_run_report.json"
DEFAULT_CSV_OUT = "outputs/labels/label_pipeline_dry_run_report.csv"


def _target_validation_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for idx, row in df.iterrows():
        future_risk = float(row.get("future_risk_label", 0.0))
        future_severity = float(row.get("future_severity", 0.0))
        weighted = float(row.get("severity_weighted_future_risk", future_risk * future_severity))
        errors = []
        if not 0.0 <= future_risk <= 1.0:
            errors.append("future_risk_label outside [0, 1]")
        if not 0.0 <= future_severity <= 1.0:
            errors.append("future_severity outside [0, 1]")
        if not 0.0 <= weighted <= 1.0:
            errors.append("severity_weighted_future_risk outside [0, 1]")
        if abs(weighted - (future_risk * future_severity)) > 1e-9:
            errors.append("severity_weighted_future_risk != future_risk_label * future_severity")
        if future_risk == 0.0 and future_severity != 0.0:
            errors.append("future_severity nonzero when future_risk_label is zero")
        rows.append(
            {
                "row_index": int(idx),
                "episode_id": row.get("episode_id"),
                "step_id": row.get("step_id"),
                "valid": not errors,
                "errors": "; ".join(errors),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_index", "episode_id", "step_id", "valid", "errors"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_label_pipeline_dry_run(
    prefix_path: str | Path = DEFAULT_PREFIX,
    json_out: str | Path = DEFAULT_JSON_OUT,
    csv_out: str | Path = DEFAULT_CSV_OUT,
) -> dict[str, Any]:
    prefix_path = Path(prefix_path)
    json_out = Path(json_out)
    csv_out = Path(csv_out)
    df = pd.read_csv(prefix_path)
    preview = df.head(20).copy()
    if "severity_weighted_future_risk" not in preview.columns:
        preview["severity_weighted_future_risk"] = preview["future_risk_label"] * preview["future_severity"]
    validation_rows = _target_validation_rows(preview)

    judge = example_risky_label()
    merged_label = merge_label_sources(
        episode_id=judge["episode_id"],
        benchmark_metadata={"risk_event": False, "severity": 0.0, "t_risk": None},
        oracle_diagnostic={"oracle_violation": True, "first_risk_step": 2},
        llm_judge=judge,
        suffix_policy="future_only",
    )
    propagation_preview = [
        {"step_id": step, **propagated_prefix_label(step, judge)}
        for step in [1, 2, 3]
    ]
    report = {
        "status": "ok" if all(row["valid"] for row in validation_rows) else "target_validation_failed",
        "input_prefix": str(prefix_path),
        "row_count_checked": len(validation_rows),
        "valid_rows": sum(1 for row in validation_rows if row["valid"]),
        "invalid_rows": sum(1 for row in validation_rows if not row["valid"]),
        "target_columns_supported": [
            "future_risk_label",
            "future_severity",
            "severity_weighted_future_risk",
        ],
        "allowed_training_targets": ["future_risk_label", "severity_weighted_future_risk"],
        "label_source_merge_preview": merged_label,
        "prefix_propagation_preview": propagation_preview,
        "outputs": {"json": str(json_out), "csv": str(csv_out)},
        "rules": {
            "will_call_provider": False,
            "will_run_agentdojo": False,
            "will_run_agenthazard": False,
            "will_train": False,
            "will_calibrate": False,
            "test_split_used": False,
        },
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_out, validation_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run judge label merge and severity target validation.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    args = parser.parse_args()
    result = run_label_pipeline_dry_run(prefix_path=args.prefix, json_out=args.json_out, csv_out=args.csv_out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

