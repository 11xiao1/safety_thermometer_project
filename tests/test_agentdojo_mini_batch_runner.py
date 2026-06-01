import argparse
import csv
import json
import os
import subprocess
import sys

from scripts import run_agentdojo_mini_batch as runner
from src.adapters.agentdojo_tools_wrapper import MaxToolCallsExceeded


SCRIPT = "scripts/run_agentdojo_mini_batch.py"


def _run(args, env=None):
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_select_tasks_uses_numeric_order_and_limit():
    available = ["user_task_0", "user_task_1", "user_task_10", "user_task_2", "user_task_3"]

    assert runner.select_tasks(available, limit=4) == [
        "user_task_0",
        "user_task_1",
        "user_task_2",
        "user_task_3",
    ]


def test_select_tasks_can_start_at_task_id():
    available = ["user_task_0", "user_task_1", "user_task_2", "user_task_3"]

    assert runner.select_tasks(available, limit=2, task_start="user_task_2") == [
        "user_task_2",
        "user_task_3",
    ]


def test_select_tasks_respects_explicit_task_list():
    available = ["user_task_0", "user_task_1", "user_task_2"]

    assert runner.select_tasks(available, limit=2, tasks="user_task_2,user_task_0") == [
        "user_task_2",
        "user_task_0",
    ]


def test_dry_run_lists_selected_tasks_and_does_not_call_provider():
    result = _run([
        "--dry-run",
        "--suite",
        "workspace",
        "--limit",
        "2",
        "--provider",
        "openai-compatible",
        "--model",
        "gpt-test",
    ])

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "dry_run"
    assert payload["selected_tasks"] == ["user_task_0", "user_task_1"]
    assert payload["will_call_provider"] is False
    assert payload["will_run_full_benchmark"] is False
    assert "proxy.example" not in result.stdout


def test_first_real_run_command_shape_dry_run_preflight():
    result = _run([
        "--dry-run",
        "--suite",
        "workspace",
        "--limit",
        "5",
        "--task-start",
        "user_task_0",
        "--trace-dir",
        "outputs\\agentdojo_mini_batch\\traces",
        "--prefix-dir",
        "outputs\\agentdojo_mini_batch\\prefix",
        "--merged-out",
        "outputs\\agentdojo_mini_batch\\merged\\workspace_mini_batch_prefix_dataset.csv",
        "--provider",
        "openai-compatible",
        "--model",
        "gpt-3.5-turbo",
        "--max-steps",
        "3",
        "--max-tool-calls",
        "3",
        "--max-output-tokens",
        "512",
        "--temperature",
        "0",
        "--cost-guard",
    ])

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["selected_tasks"] == [
        "user_task_0",
        "user_task_1",
        "user_task_2",
        "user_task_3",
        "user_task_4",
    ]
    assert payload["outputs"]["user_task_0"]["trace"].endswith("workspace_user_task_0_trace.jsonl")
    assert payload["outputs"]["user_task_0"]["prefix"].endswith("workspace_user_task_0_prefix_dataset.csv")
    assert payload["merged_out"] == "outputs\\agentdojo_mini_batch\\merged\\workspace_mini_batch_prefix_dataset.csv"
    assert payload["summary"] == "outputs/agentdojo_mini_batch/run_summary.json"
    assert payload["cost_guard"] == {
        "enabled": True,
        "max_steps": 3,
        "max_tool_calls": 3,
        "max_output_tokens": 512,
        "temperature": 0.0,
    }
    assert payload["will_call_provider"] is False


def test_non_dry_run_requires_allow_real_run_before_provider_call():
    result = _run(["--limit", "1", "--provider", "openai-compatible", "--model", "gpt-test"])

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "blocked"
    assert "--allow-real-run" in payload["reason"]


def test_non_dry_run_requires_provider_call_opt_in():
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "secret-test-key"
    env["OPENAI_BASE_URL"] = "https://proxy.example.test/v1"
    result = _run(
        ["--limit", "1", "--allow-real-run", "--provider", "openai-compatible", "--model", "gpt-test"],
        env=env,
    )

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "blocked"
    assert "--allow-provider-call" in payload["reason"]
    assert "secret-test-key" not in result.stderr
    assert "https://proxy.example.test" not in result.stderr


def test_limit_above_ten_is_blocked_in_dry_run():
    result = _run(["--dry-run", "--limit", "11"])

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "blocked"
    assert "--limit must not exceed 10" in payload["reason"]


def test_merge_prefix_files_keeps_single_header_and_counts_rows(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    merged = tmp_path / "merged.csv"
    for path, episode_id in [(first, "episode_a"), (second, "episode_b")]:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode_id", "step_id", "future_risk_label"])
            writer.writerow([episode_id, 1, 0])
            writer.writerow([episode_id, 2, 1])

    shape = runner._merge_prefix_files([str(first), str(second)], merged)

    rows = list(csv.reader(merged.open("r", encoding="utf-8", newline="")))
    assert rows[0] == ["episode_id", "step_id", "future_risk_label"]
    assert rows.count(["episode_id", "step_id", "future_risk_label"]) == 1
    assert shape == {"rows": 4, "columns": 3}


def test_run_mini_batch_with_fake_task_runner_writes_summary_and_outputs(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    prefix_dir = tmp_path / "prefixes"
    merged_out = tmp_path / "merged" / "prefix.csv"
    merged_trace = tmp_path / "merged" / "trace.jsonl"
    summary = tmp_path / "summary.json"

    monkeypatch.setattr(runner, "_available_user_tasks", lambda args: ["user_task_0", "user_task_1"])

    class FakePlan:
        def to_dict(self):
            return {"status": "fake"}

    monkeypatch.setattr(runner, "inspect_agentdojo_pipeline", lambda package_name="agentdojo": FakePlan())

    def fake_run_one_task(args, task_id, trace_path, prefix_path, plan):
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"episode_id": task_id, "hook_type": "final"}) + "\n")
        with open(prefix_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode_id", "step_id", "future_risk_label"])
            writer.writerow([task_id, 1, 0])
        return {
            "task_id": task_id,
            "status": "ok",
            "trace": trace_path,
            "prefix": prefix_path,
            "trace_event_count": 1,
            "prefix_rows": 1,
            "prefix_columns": 3,
            "utility": True,
            "security": True,
        }

    monkeypatch.setattr(runner, "_run_one_task", fake_run_one_task)
    args = argparse.Namespace(
        suite="workspace",
        limit=2,
        task_start=None,
        tasks=None,
        trace_dir=str(trace_dir),
        prefix_dir=str(prefix_dir),
        merged_out=str(merged_out),
        merged_trace=str(merged_trace),
        summary=str(summary),
        provider="local",
        model="fake-local",
        benchmark_version="v1.2.2",
        max_steps=3,
        max_tool_calls=3,
        max_output_tokens=512,
        temperature=0,
        allow_real_run=True,
        allow_provider_call=True,
        cost_guard=True,
        dry_run=False,
        agentdojo_package="agentdojo",
    )

    payload = runner.run_mini_batch(args)

    assert payload["status"] == "ok"
    assert payload["completed_tasks"] == ["user_task_0", "user_task_1"]
    assert payload["merged_trace_rows"] == 2
    assert payload["merged_prefix_rows"] == 2
    assert summary.exists()
    assert "fake-local" in summary.read_text(encoding="utf-8")


def test_run_mini_batch_records_max_tool_calls_stop_reason(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    prefix_dir = tmp_path / "prefixes"
    merged_out = tmp_path / "merged" / "prefix.csv"
    merged_trace = tmp_path / "merged" / "trace.jsonl"
    summary = tmp_path / "summary.json"

    monkeypatch.setattr(runner, "_available_user_tasks", lambda args: ["user_task_0"])

    class FakePlan:
        def to_dict(self):
            return {"status": "fake"}

    monkeypatch.setattr(runner, "inspect_agentdojo_pipeline", lambda package_name="agentdojo": FakePlan())
    monkeypatch.setattr(runner, "_tool_call_count_from_trace", lambda trace_path: 3)

    def fake_run_one_task(args, task_id, trace_path, prefix_path, plan):
        raise MaxToolCallsExceeded(max_tool_calls=3, tool_calls_seen=3)

    monkeypatch.setattr(runner, "_run_one_task", fake_run_one_task)
    args = argparse.Namespace(
        suite="workspace",
        limit=1,
        task_start=None,
        tasks=None,
        trace_dir=str(trace_dir),
        prefix_dir=str(prefix_dir),
        merged_out=str(merged_out),
        merged_trace=str(merged_trace),
        summary=str(summary),
        provider="local",
        model="fake-local",
        benchmark_version="v1.2.2",
        max_steps=3,
        max_tool_calls=3,
        max_output_tokens=512,
        temperature=0,
        allow_real_run=True,
        allow_provider_call=True,
        cost_guard=True,
        dry_run=False,
        agentdojo_package="agentdojo",
    )

    payload = runner.run_mini_batch(args)

    assert payload["status"] == "stopped"
    assert payload["failed_tasks"] == ["user_task_0"]
    result = payload["task_results"][0]
    assert result["status"] == "stopped"
    assert result["stop_reason"] == "max_tool_calls_exceeded"
    assert result["tool_call_count"] == 3
    assert result["max_tool_calls"] == 3
    assert result["max_tool_calls_hit"] is True
    assert json.loads(summary.read_text(encoding="utf-8"))["task_results"][0]["stop_reason"] == "max_tool_calls_exceeded"
