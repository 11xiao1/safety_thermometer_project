import json

from scripts.agenthazard.audit_agenthazard_trace_events import audit_agenthazard_trace_events


def _event(episode_id, step_id, hook_type, notes, user_instruction="Do task", proposed_tool=None, observation=None):
    return {
        "episode_id": episode_id,
        "step_id": step_id,
        "hook_type": hook_type,
        "user_instruction": user_instruction,
        "proposed_tool": proposed_tool,
        "tool_args": {},
        "observation": observation,
        "state_delta": {},
        "self_check": {},
        "notes": json.dumps(notes, sort_keys=True),
    }


def test_audit_agenthazard_trace_events_reports_counts_and_metadata_join(tmp_path):
    trace = tmp_path / "events.jsonl"
    notes = {
        "benchmark_name": "agenthazard",
        "agent_family": "claudecode",
        "task_id": 7,
        "source_batch": "traces/claudecode/mock.zip",
        "source_trace_path": "mock/trajectory_7.jsonl",
    }
    rows = [
        _event("agenthazard:claudecode:mock:System_RCE:7", 1, "pre_step", notes, proposed_tool="Read"),
        _event("agenthazard:claudecode:mock:System_RCE:7", 1, "pre_step", notes, proposed_tool="Read"),
        _event("agenthazard:claudecode:mock:System_RCE:7", 2, "final", notes, observation="done"),
        "{bad json",
    ]
    trace.write_text("\n".join(json.dumps(row) if isinstance(row, dict) else row for row in rows), encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"event_count": 3, "trajectory_file_count": 1}), encoding="utf-8")
    metadata = tmp_path / "dataset.json"
    metadata.write_text(json.dumps([{"id": 7, "category": "System_RCE", "jailbreak_method": "Direct", "query": "Do task"}]), encoding="utf-8")

    result = audit_agenthazard_trace_events(
        trace_events_jsonl=trace,
        conversion_summary_json=summary,
        metadata_json=metadata,
        audit_json=tmp_path / "audit.json",
        audit_csv=tmp_path / "audit.csv",
    )

    assert result["scalar_counts"]["total_event_count"] == 3
    assert result["scalar_counts"]["total_episode_count"] == 1
    assert result["scalar_counts"]["total_task_count"] == 1
    assert result["scalar_counts"]["malformed_jsonl_line_count"] == 1
    assert result["scalar_counts"]["duplicate_semantic_event_count"] == 1
    assert result["scalar_counts"]["converted_event_count_matches_summary"] == 1
    assert result["events_by_agent_family"] == {"claudecode": 3}
    assert result["metadata_join_coverage"]["matched_tasks"] == 1
    assert result["labels"]["label_status"] == "missing_pending_judge"
    assert (tmp_path / "audit.csv").exists()
