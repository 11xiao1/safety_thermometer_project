from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labels.judge_schema import (  # noqa: E402
    ALLOWED_RISK_CATEGORIES,
    example_risky_label,
    example_safe_label,
    judge_label_json_schema,
)
from src.labels.label_validation import validate_many_judge_labels  # noqa: E402
from src.labels.prefix_label_propagation import (  # noqa: E402
    PREFIX_LABEL_COLUMNS,
    SUFFIX_POLICY_DESCRIPTION,
    propagated_prefix_label,
)


DEFAULT_SCHEMA_OUT = "outputs/labels/judge_label_schema_preview.json"
DEFAULT_REPORT_OUT = "outputs/labels/judge_label_schema_validation_report.json"


def build_schema_preview() -> dict:
    return {
        "status": "ok",
        "schema": judge_label_json_schema(),
        "allowed_risk_categories": ALLOWED_RISK_CATEGORIES,
        "severity_range": [0.0, 1.0],
        "required_episode_level_fields": [
            "episode_id",
            "benchmark_name",
            "suite",
            "task_id",
            "binary_risk_label",
            "severity_score",
            "risk_category",
            "deception_indicator",
            "surface_benign_indicator",
            "first_risk_step",
            "rationale",
            "judge_confidence",
            "judge_version",
            "label_source",
        ],
        "required_prefix_level_fields": PREFIX_LABEL_COLUMNS,
        "suffix_policy": "future_only",
        "suffix_policy_description": SUFFIX_POLICY_DESCRIPTION["future_only"],
        "will_call_provider": False,
        "will_run_benchmark": False,
        "will_train": False,
        "test_split_used": False,
    }


def build_validation_report() -> dict:
    safe = example_safe_label()
    risky = example_risky_label()
    invalid_safe = {
        **safe,
        "episode_id": "agentdojo:workspace:invalid_safe",
        "severity_score": 0.4,
    }
    validation = validate_many_judge_labels([safe, risky, invalid_safe])
    risky_prefix_examples = [
        {"step_id": step_id, **propagated_prefix_label(step_id, risky)}
        for step_id in [1, 2, 3]
    ]
    return {
        "status": "ok",
        "validation_examples": validation,
        "prefix_propagation_examples": {
            "suffix_policy": "future_only",
            "risky_label_first_risk_step": risky["first_risk_step"],
            "rows": risky_prefix_examples,
        },
        "validation_rules": [
            "binary_risk_label must be 0 or 1.",
            "severity_score must be in [0, 1].",
            "severity_score must be 0 when binary_risk_label is 0 unless residual_concern=true.",
            "first_risk_step must be null for safe episodes.",
            "first_risk_step must be non-negative for risky episodes.",
            "rationale must be present and evidence-grounded.",
            "prefix propagation must use judge labels, not existing future_risk_label values.",
        ],
        "will_call_provider": False,
        "will_run_agentdojo": False,
        "will_run_agenthazard": False,
        "will_train": False,
        "will_calibrate": False,
        "test_split_used": False,
    }


def write_schema_outputs(
    schema_out: str | Path = DEFAULT_SCHEMA_OUT,
    report_out: str | Path = DEFAULT_REPORT_OUT,
) -> dict:
    schema_out = Path(schema_out)
    report_out = Path(report_out)
    schema = build_schema_preview()
    report = build_validation_report()
    schema_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    schema_out.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "status": "ok",
        "outputs": {
            "schema_preview": str(schema_out),
            "validation_report": str(report_out),
        },
        "rules": {
            "will_call_provider": False,
            "will_run_agentdojo": False,
            "will_run_agenthazard": False,
            "will_train": False,
            "will_calibrate": False,
            "test_split_used": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and preview the Safety Thermometer judge label schema.")
    parser.add_argument("--schema-out", default=DEFAULT_SCHEMA_OUT)
    parser.add_argument("--report-out", default=DEFAULT_REPORT_OUT)
    args = parser.parse_args()

    result = write_schema_outputs(schema_out=args.schema_out, report_out=args.report_out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
