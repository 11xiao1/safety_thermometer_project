import json
import zipfile

from scripts.agenthazard.inspect_agenthazard_trace_archives import inspect_agenthazard_trace_archives
from src.adapters.agenthazard_adapter import AgentHazardAdapter


def _write_zip(path, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)


def test_agenthazard_adapter_inspects_mock_trace_archive(tmp_path):
    repo = tmp_path / "AgentHazard"
    archive_path = repo / "traces" / "openclaw" / "mock.zip"
    _write_zip(
        archive_path,
        {
            "session/history.jsonl": "\n".join(
                [
                    json.dumps(
                        {
                            "task_id": "task_1",
                            "step_id": 0,
                            "command": "ls",
                            "tool_args": {"path": "."},
                            "observation": "ok",
                            "timestamp": "2026-01-01T00:00:00Z",
                        }
                    )
                ]
            ),
            "results/outcome.json": json.dumps({"task_id": "task_1", "harmful": False, "success": True}),
            "metadata/task.json": json.dumps({"query": "List files"}),
        },
    )

    adapter = AgentHazardAdapter(repo)
    report = adapter.inspect_trace_archive(archive_path)

    assert report["agent_family"] == "openclaw"
    assert report["internal_file_count"] == 3
    assert "session/history.jsonl" in report["candidate_trajectory_files"]
    capabilities = report["detected_fields"]["detected_capabilities"]
    assert capabilities["task_id"] is True
    assert capabilities["step_id"] is True
    assert capabilities["command"] is True
    assert capabilities["observation"] is True
    assert report["schema_signature"].startswith("openclaw|")


def test_inspect_agenthazard_trace_archives_writes_reports_and_plan(tmp_path):
    repo = tmp_path / "AgentHazard"
    _write_zip(
        repo / "traces" / "iflow" / "mock.zip",
        {
            "trace/steps.json": json.dumps(
                [
                    {
                        "instance_id": "case_2",
                        "step": 1,
                        "tool_call": {"name": "bash"},
                        "arguments": {"cmd": "pwd"},
                        "result": "workspace",
                        "label": "safe",
                    }
                ]
            )
        },
    )

    result = inspect_agenthazard_trace_archives(
        repo_path=repo,
        schema_json=tmp_path / "outputs/agenthazard/schema.json",
        schema_csv=tmp_path / "outputs/agenthazard/schema.csv",
        plan_json=tmp_path / "outputs/agenthazard/plan.json",
    )
    schema = json.loads((tmp_path / "outputs/agenthazard/schema.json").read_text(encoding="utf-8"))
    plan = json.loads((tmp_path / "outputs/agenthazard/plan.json").read_text(encoding="utf-8"))

    assert result["archive_count"] == 1
    assert result["conversion_feasible"] is True
    assert schema["schema_family_count"] == 1
    assert plan["schema_families"][0]["agent_families"] == ["iflow"]
    assert (tmp_path / "outputs/agenthazard/schema.csv").exists()
