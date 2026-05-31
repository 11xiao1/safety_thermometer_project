import json
import subprocess
import sys


SCRIPT = "scripts/run_agentdojo_smoke_trace.py"


def _run(args):
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_agentdojo_smoke_dry_run_runs_without_api_key():
    result = _run([
        "--suite",
        "workspace",
        "--user-task",
        "user_task_0",
        "--trace",
        "outputs/agentdojo_smoke_trace.jsonl",
        "--dry-run",
    ])

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "dry_run"
    assert payload["will_call_provider"] is False


def test_agentdojo_smoke_dry_run_does_not_call_provider():
    result = _run(["--dry-run", "--provider", "openai", "--model", "gpt-test"])

    payload = json.loads(result.stdout)
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-test"
    assert payload["will_call_provider"] is False
    assert payload["will_run_benchmark"] is False


def test_agentdojo_smoke_dry_run_outputs_structured_plan():
    result = _run(["--dry-run"])

    payload = json.loads(result.stdout)
    assert "agentdojo" in payload
    assert "replacement" in payload["agentdojo"]
    assert payload["will_monkey_patch"] is False


def test_agentdojo_smoke_dry_run_reports_unavailable_agentdojo_without_crashing():
    result = _run(["--dry-run", "--agentdojo-package", "definitely_not_installed_agentdojo"])

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["agentdojo"]["status"] == "AgentDojo not installed"
    assert payload["agentdojo"]["agentdojo_installed"] is False


def test_agentdojo_smoke_non_dry_run_fails_safely_until_wired():
    result = _run(["--suite", "workspace", "--user-task", "user_task_0"])

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "not_implemented"
    assert "TraceHookedToolsExecutor" in payload["reason"]
