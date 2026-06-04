import pandas as pd

from src.labels.judge_schema import example_risky_label, example_safe_label
from src.labels.prefix_label_propagation import (
    PREFIX_LABEL_COLUMNS,
    propagate_label_to_prefix_dataframe,
    propagated_prefix_label,
)


def test_prefix_label_propagation_uses_future_only_suffix_policy():
    label = example_risky_label()

    before = propagated_prefix_label(1, label)
    at_risk = propagated_prefix_label(2, label)
    after = propagated_prefix_label(3, label)

    assert before["future_risk_label"] == 1
    assert before["future_severity"] == label["severity_score"]
    assert before["severity_weighted_future_risk"] == label["severity_score"]
    assert at_risk["future_risk_label"] == 1
    assert after["future_risk_label"] == 0
    assert after["future_severity"] == 0.0
    assert after["severity_weighted_future_risk"] == 0.0
    assert after["risk_category"] == "none"


def test_safe_label_propagates_zero_targets():
    propagated = propagated_prefix_label(1, example_safe_label())

    assert propagated["future_risk_label"] == 0
    assert propagated["future_severity"] == 0.0
    assert propagated["severity_weighted_future_risk"] == 0.0
    assert propagated["first_risk_step"] is None


def test_propagate_label_to_prefix_dataframe_adds_required_columns_without_reading_existing_labels():
    df = pd.DataFrame(
        [
            {"episode_id": "agentdojo:workspace:user_task_1", "step_id": 1, "future_risk_label": 0},
            {"episode_id": "agentdojo:workspace:user_task_1", "step_id": 3, "future_risk_label": 1},
        ]
    )

    propagated = propagate_label_to_prefix_dataframe(df, example_risky_label())

    assert all(column in propagated.columns for column in PREFIX_LABEL_COLUMNS)
    assert propagated["future_risk_label"].tolist() == [1, 0]
    assert propagated["severity_weighted_future_risk"].tolist() == [0.8, 0.0]
