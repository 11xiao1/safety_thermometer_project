from src.labels.judge_schema import example_risky_label, example_safe_label
from src.labels.label_merger import merge_label_sources


def test_label_merger_prefers_judge_and_reports_metadata_conflict():
    judge = example_risky_label()

    merged = merge_label_sources(
        episode_id=judge["episode_id"],
        benchmark_metadata={"risk_event": False, "severity": 0.0, "t_risk": None},
        oracle_diagnostic={"oracle_violation": True, "first_risk_step": 2},
        llm_judge=judge,
    )

    assert merged["label_source"] == "llm_judge"
    assert merged["binary_risk_label"] == 1
    assert merged["label_conflict_status"] == "conflict"
    assert "judge risky but metadata safe" in merged["conflicts"]
    assert merged["available_sources"]["oracle_diagnostic"] is True
    assert merged["oracle_used_as_label"] is False


def test_label_merger_detects_metadata_risky_judge_safe():
    judge = example_safe_label()

    merged = merge_label_sources(
        episode_id=judge["episode_id"],
        benchmark_metadata={"risk_event": True, "severity": 0.6, "t_risk": 1},
        llm_judge=judge,
    )

    assert merged["label_source"] == "llm_judge"
    assert merged["label_conflict_status"] == "conflict"
    assert "metadata risky but judge safe" in merged["conflicts"]


def test_label_merger_reports_invalid_judge_without_silent_resolution():
    judge = {**example_safe_label(), "severity_score": 0.5}

    merged = merge_label_sources(
        episode_id=judge["episode_id"],
        benchmark_metadata=None,
        llm_judge=judge,
    )

    assert merged["label_source"] is None
    assert merged["label_conflict_status"] == "conflict"
    assert merged["conflicts"]

