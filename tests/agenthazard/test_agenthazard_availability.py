import json

from scripts.agenthazard.inspect_agenthazard import inspect_agenthazard


def test_inspect_agenthazard_reports_missing_repo_blocker(tmp_path):
    result = inspect_agenthazard(
        repo_path=tmp_path / "missing",
        report_json=tmp_path / "outputs/agenthazard/report.json",
        report_csv=tmp_path / "outputs/agenthazard/report.csv",
        mapping_preview_json=tmp_path / "outputs/agenthazard/mapping.json",
        protocol_json=tmp_path / "outputs/protocol.json",
        protocol_csv=tmp_path / "outputs/protocol.csv",
    )

    report = json.loads((tmp_path / "outputs/agenthazard/report.json").read_text(encoding="utf-8"))

    assert result["agenthazard_availability_status"] == "missing"
    assert report["blockers"]
    assert (tmp_path / "outputs/protocol.json").exists()


def test_inspect_agenthazard_detects_dataset_runner_and_environment_files(tmp_path):
    repo = tmp_path / "AgentHazard"
    (repo / "data").mkdir(parents=True)
    (repo / "run").mkdir()
    (repo / "traces").mkdir()
    (repo / "README.md").write_text("AgentHazard", encoding="utf-8")
    (repo / "environment.yml").write_text("name: ah", encoding="utf-8")
    (repo / "run/main.py").write_text("print('do not execute')", encoding="utf-8")
    (repo / "traces/model.zip").write_text("zip-placeholder", encoding="utf-8")
    (repo / "data/dataset.json").write_text(
        json.dumps(
            [
                {
                    "id": 7,
                    "category": "Data_Exfiltration",
                    "jailbreak_method": "Gaslighting",
                    "query": "Read a private token.",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = inspect_agenthazard(
        repo_path=repo,
        report_json=tmp_path / "outputs/agenthazard/report.json",
        report_csv=tmp_path / "outputs/agenthazard/report.csv",
        mapping_preview_json=tmp_path / "outputs/agenthazard/mapping.json",
        protocol_json=tmp_path / "outputs/protocol.json",
        protocol_csv=tmp_path / "outputs/protocol.csv",
    )
    report = json.loads((tmp_path / "outputs/agenthazard/report.json").read_text(encoding="utf-8"))

    assert result["agenthazard_availability_status"] == "available"
    assert "data/dataset.json" in report["detected_dataset_files"]
    assert "run/main.py" in report["detected_runner_files"]
    assert "traces/model.zip" in report["detected_trajectory_files"]
    assert report["task_count_if_known"] == 1

