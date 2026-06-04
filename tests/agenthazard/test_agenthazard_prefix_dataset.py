import csv
import json

from scripts.agenthazard.build_agenthazard_prefix_dataset import build_agenthazard_prefix_dataset_preview


def test_build_agenthazard_prefix_dataset_preview_joins_metadata_and_marks_labels_pending(tmp_path):
    trace = tmp_path / "events.jsonl"
    notes = {
        "benchmark_name": "agenthazard",
        "benchmark_role": "primary",
        "agent_family": "iflow",
        "task_id": 9,
        "source_batch": "traces/iflow/mock.zip",
        "source_trace_path": "mock/trajectory_9.jsonl",
    }
    event = {
        "episode_id": "agenthazard:iflow:mock:Data_Exfiltration:9",
        "step_id": 1,
        "hook_type": "checkpoint",
        "user_instruction": "Read a file",
        "plan_summary": "Read private token",
        "proposed_tool": None,
        "tool_args": {},
        "observation": "done",
        "state_delta": {},
        "self_check": {},
        "notes": json.dumps(notes, sort_keys=True),
    }
    trace.write_text(json.dumps(event), encoding="utf-8")
    metadata = tmp_path / "dataset.json"
    metadata.write_text(
        json.dumps(
            [
                {
                    "id": 9,
                    "category": "Data_Exfiltration",
                    "jailbreak_method": "Direct",
                    "query": "Read a file",
                    "harmful": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = build_agenthazard_prefix_dataset_preview(
        trace_events_jsonl=trace,
        metadata_json=metadata,
        prefix_csv=tmp_path / "prefix.csv",
        manifest_json=tmp_path / "manifest.json",
        max_rows=10,
    )
    rows = list(csv.DictReader((tmp_path / "prefix.csv").open(encoding="utf-8")))

    assert result["prefix_rows"] == 1
    assert result["metadata_join_coverage"]["matched_tasks_in_preview"] == 1
    assert result["labels"]["label_status"] == "missing_pending_judge"
    assert rows[0]["benchmark_name"] == "agenthazard"
    assert rows[0]["benchmark_role"] == "primary"
    assert rows[0]["source_agent_family"] == "iflow"
    assert rows[0]["risk_category"] == "Data_Exfiltration"
    assert rows[0]["attack_strategy"] == "Direct"
    assert rows[0]["future_risk_label"] == ""
    assert rows[0]["label_status"] == "missing_pending_judge"
