import pandas as pd

from src.features.extractor import FEATURE_COLUMNS
from src.monitor.replay import make_prefix_dataset
from src.monitor.schema import TraceEvent


TRACE_PATH = "data/samples/toy_episodes.jsonl"
CUMULATIVE_COLUMNS = [
    "cumulative_state_modifying_count",
    "cumulative_irreversible_count",
    "cumulative_sensitive_access_count",
    "cumulative_external_send_count",
    "cumulative_fallback_count",
    "confirmation_seen_so_far",
    "max_risk_score_so_far",
]
REQUIRED_COLUMNS = [
    "episode_id",
    "step_id",
    "hook_type",
    *FEATURE_COLUMNS,
    *CUMULATIVE_COLUMNS,
    "oracle_violation",
    "oracle_rules",
    "risk_score",
    "policy",
    "future_risk_label",
    "future_severity",
    "t_risk",
    "lead_time_if_alert_now",
]


def _event(**overrides) -> TraceEvent:
    defaults = {
        "episode_id": "lead_time_unit",
        "step_id": 1,
        "hook_type": "pre_step",
        "user_instruction": "",
        "plan_summary": "",
        "proposed_tool": None,
        "tool_args": {},
        "observation": None,
        "state_delta": {},
        "self_check": {},
        "severity": 0.0,
    }
    defaults.update(overrides)
    return TraceEvent(**defaults)


def _write_trace(path, events: list[TraceEvent]) -> None:
    path.write_text(
        "\n".join(event.model_dump_json() for event in events) + "\n",
        encoding="utf-8",
    )


def test_prefix_dataset_contains_required_columns():
    df = make_prefix_dataset(TRACE_PATH)

    assert all(column in df.columns for column in REQUIRED_COLUMNS)


def test_cumulative_fallback_count_increases_after_fallback():
    df = make_prefix_dataset(TRACE_PATH)
    episode = df[df["episode_id"] == "toy_unprotected_fallback"]

    assert episode["cumulative_fallback_count"].tolist() == [0, 1]


def test_confirmation_seen_so_far_remains_false_for_implicit_confirmation():
    df = make_prefix_dataset(TRACE_PATH)
    episode = df[df["episode_id"] == "toy_email_without_confirmation"]

    assert episode["confirmation_seen_so_far"].tolist() == [0]


def test_max_risk_score_so_far_is_monotonic_within_each_episode():
    df = make_prefix_dataset(TRACE_PATH)

    for _, episode in df.groupby("episode_id"):
        assert episode["max_risk_score_so_far"].is_monotonic_increasing


def test_lead_time_is_positive_zero_and_negative_around_risky_step(tmp_path):
    trace_path = tmp_path / "lead_time_trace.jsonl"
    _write_trace(
        trace_path,
        [
            _event(
                step_id=1,
                user_instruction="Inspect the public document first.",
                plan_summary="Read only.",
                proposed_tool="read_file",
                tool_args={"path": "public.md"},
                t_risk=2,
            ),
            _event(
                step_id=2,
                user_instruction="Delete Doc B.",
                plan_summary="Confirmed by benchmark metadata as the risky step.",
                proposed_tool="read_file",
                tool_args={"path": "public.md"},
                risk_event=True,
                risk_type="benchmark_risk",
                t_risk=2,
                severity=0.8,
            ),
            _event(
                step_id=3,
                user_instruction="Summarize the result.",
                plan_summary="Read only after the risky step.",
                proposed_tool="read_file",
                tool_args={"path": "public.md"},
                t_risk=2,
                severity=0.0,
            ),
        ],
    )

    df = make_prefix_dataset(str(trace_path))

    assert df["lead_time_if_alert_now"].tolist() == [1, 0, -1]


def test_safe_episode_has_no_future_risk_label():
    df = make_prefix_dataset(TRACE_PATH)
    episode = df[df["episode_id"] == "toy_safe_read"]

    assert episode["future_risk_label"].tolist() == [0, 0]
