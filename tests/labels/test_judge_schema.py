from src.labels.judge_schema import (
    ALLOWED_RISK_CATEGORIES,
    JudgeLabel,
    example_risky_label,
    example_safe_label,
    judge_label_json_schema,
)
from src.labels.label_validation import validation_errors


def test_judge_schema_accepts_safe_and_risky_examples():
    safe = JudgeLabel.model_validate(example_safe_label())
    risky = JudgeLabel.model_validate(example_risky_label())

    assert safe.binary_risk_label == 0
    assert safe.severity_score == 0.0
    assert safe.first_risk_step is None
    assert risky.binary_risk_label == 1
    assert risky.first_risk_step == 2
    assert risky.risk_category in ALLOWED_RISK_CATEGORIES


def test_judge_schema_rejects_inconsistent_safe_label():
    payload = {
        **example_safe_label(),
        "severity_score": 0.3,
    }

    errors = validation_errors(payload)

    assert errors
    assert any("severity_score=0" in error for error in errors)


def test_judge_schema_requires_evidence_grounded_rationale():
    payload = {
        **example_risky_label(),
        "rationale": "Bad outcome.",
    }

    errors = validation_errors(payload)

    assert errors
    assert any("rationale" in error for error in errors)


def test_judge_json_schema_is_strict_and_lists_required_fields():
    schema = judge_label_json_schema()

    assert schema["additionalProperties"] is False
    assert "episode_id" in schema["required"]
    assert schema["x_allowed_risk_categories"] == ALLOWED_RISK_CATEGORIES
    assert schema["x_severity_range"] == [0.0, 1.0]
