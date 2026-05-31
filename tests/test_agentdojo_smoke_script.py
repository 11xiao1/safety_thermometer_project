import json
import os
import subprocess
import sys


SCRIPT = "scripts/run_agentdojo_smoke_trace.py"


def _run(args, env=None):
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
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
    result = _run(["--dry-run", "--provider", "openai-compatible", "--model", "gpt-test"])

    payload = json.loads(result.stdout)
    assert payload["provider"]["provider"] == "openai-compatible"
    assert payload["provider"]["model"] == "gpt-test"
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
    assert payload["status"] == "blocked"
    assert "--allow-real-run" in payload["reason"]


def test_agentdojo_smoke_list_suites_requires_no_api_key():
    result = _run(["--list-suites", "--agentdojo-package", "definitely_not_installed_agentdojo"])

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "AgentDojo not installed"


def test_agentdojo_smoke_list_tasks_requires_no_api_key():
    result = _run([
        "--list-tasks",
        "--suite",
        "workspace",
        "--agentdojo-package",
        "definitely_not_installed_agentdojo",
    ])

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "AgentDojo not installed"


def test_openai_compatible_real_run_fails_clearly_without_env_vars():
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_BASE_URL", None)
    result = _run(["--allow-real-run", "--provider", "openai-compatible", "--model", "gpt-test"], env=env)

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "blocked"
    assert payload["reason"] == "Missing required OpenAI-compatible provider environment variables."
    assert "OPENAI_API_KEY" in payload["missing_env"]
    assert "OPENAI_BASE_URL" in payload["missing_env"]


def test_openai_compatible_does_not_print_api_key():
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "secret-test-key"
    env["OPENAI_BASE_URL"] = "https://proxy.example.test/v1"
    result = _run(
        ["--allow-real-run", "--provider", "openai-compatible", "--model", "gpt-test"],
        env=env,
    )

    combined = result.stdout + result.stderr
    assert "secret-test-key" not in combined
    assert "openai_api_key_present" in combined


def test_agentdojo_smoke_allow_real_run_fails_safely_if_wiring_incomplete():
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "secret-test-key"
    env["OPENAI_BASE_URL"] = "https://proxy.example.test/v1"
    result = _run(
        ["--allow-real-run", "--provider", "openai-compatible", "--model", "gpt-test"],
        env=env,
    )

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "not_implemented"
    assert "TraceHookedToolsExecutor" in payload["reason"]


def test_max_steps_and_max_tool_calls_are_parsed_and_reported():
    result = _run(["--dry-run", "--max-steps", "2", "--max-tool-calls", "2", "--max-output-tokens", "64"])

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["cost_guard"]["max_steps"] == 2
    assert payload["cost_guard"]["max_tool_calls"] == 2
    assert payload["cost_guard"]["max_output_tokens"] == 64
