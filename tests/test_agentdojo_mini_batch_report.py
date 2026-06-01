import csv
import json
import subprocess
import sys

from scripts.report_agentdojo_mini_batch import build_report, write_report


SCRIPT = "scripts/report_agentdojo_mini_batch.py"


def _write_mock_inputs(tmp_path):
    summary_path = tmp_path / "run_summary.json"
    trace_path = tmp_path / "workspace_mini_batch_trace.jsonl"
    prefix_path = tmp_path / "workspace_mini_batch_prefix_dataset.csv"

    summary = {
        "status": "stopped",
        "selected_tasks": ["user_task_0", "user_task_1", "user_task_2"],
        "completed_tasks": ["user_task_0", "user_task_1"],
        "failed_tasks": ["user_task_2"],
        "skipped_tasks": [],
        "task_results": [
            {
                "task_id": "user_task_0",
                "status": "ok",
                "utility": True,
                "security": True,
            },
            {
                "task_id": "user_task_1",
                "status": "ok",
                "utility": False,
                "security": True,
            },
            {
                "task_id": "user_task_2",
                "status": "failed",
                "error": "RuntimeError: stopped by guard",
            },
        ],
        "cost_guard": {
            "max_steps": 3,
            "max_tool_calls": 3,
            "max_output_tokens": 512,
            "temperature": 0,
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    events = [
        {"hook_type": "pre_step", "proposed_tool": "search_calendar_events"},
        {"hook_type": "post_step", "proposed_tool": "search_calendar_events"},
        {"hook_type": "pre_step", "proposed_tool": "read_email"},
        {"hook_type": "post_step", "proposed_tool": "read_email"},
        {"hook_type": "final"},
    ]
    trace_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    with prefix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_id", "step_id", "future_risk_label", "oracle_violation"])
        writer.writerow(["episode_a", 1, 0, 0])
        writer.writerow(["episode_a", 2, 1, 0])
        writer.writerow(["episode_b", 1, 0, 1])

    return summary_path, trace_path, prefix_path


def test_build_report_summarizes_mock_mini_batch_outputs(tmp_path):
    summary_path, trace_path, prefix_path = _write_mock_inputs(tmp_path)

    report = build_report(summary_path, trace_path, prefix_path)

    assert "This is a mini-batch smoke report, not a benchmark-scale result." in report
    assert "Completed task count: 2" in report
    assert "Failed task count: 1" in report
    assert "`user_task_2`: `failed`" in report
    assert "TraceEvent rows: 5" in report
    assert "`pre_step`: 2" in report
    assert "Tool call count: 2" in report
    assert "`read_email`" in report
    assert "Merged prefix rows: 3" in report
    assert "Merged prefix columns: 4" in report
    assert "`future_risk_label` distribution: `0`: 2, `1`: 1" in report
    assert "`oracle_violation` distribution: `0`: 2, `1`: 1" in report
    assert "`max_output_tokens`: 512" in report
    assert "No calibration claim on AgentDojo yet." in report


def test_write_report_creates_markdown_file(tmp_path):
    summary_path, trace_path, prefix_path = _write_mock_inputs(tmp_path)
    out_path = tmp_path / "report.md"

    written = write_report(summary_path, trace_path, prefix_path, out_path)

    assert written == str(out_path)
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").startswith("# AgentDojo Mini-Batch Report")


def test_report_cli_runs_on_mock_outputs(tmp_path):
    summary_path, trace_path, prefix_path = _write_mock_inputs(tmp_path)
    out_path = tmp_path / "report.md"

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--summary",
            str(summary_path),
            "--trace",
            str(trace_path),
            "--prefix",
            str(prefix_path),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert out_path.exists()


def test_report_cli_missing_input_fails_clearly(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--summary",
            str(tmp_path / "missing_summary.json"),
            "--trace",
            str(tmp_path / "missing_trace.jsonl"),
            "--prefix",
            str(tmp_path / "missing_prefix.csv"),
            "--out",
            str(tmp_path / "report.md"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "blocked"
    assert "Missing run summary" in payload["reason"]
    assert payload["will_call_provider"] is False
    assert payload["will_run_agentdojo"] is False
