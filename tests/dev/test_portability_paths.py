import subprocess

from scripts.dev.check_portability import build_portability_report


def test_portability_report_passes_core_ignore_rules(tmp_path):
    (tmp_path / ".gitignore").write_text("outputs/\nexternal/AgentHazard/\n", encoding="utf-8")
    (tmp_path / "requirements-freeze.txt").write_text("pandas>=2\n", encoding="utf-8")
    script = tmp_path / "scripts/agenthazard/convert_agenthazard_traces.py"
    script.parent.mkdir(parents=True)
    script.write_text("import argparse\nfrom pathlib import Path\nPath('.')\n", encoding="utf-8")

    report = build_portability_report(tmp_path)

    assert report["status"] in {"ready_with_warnings", "blocked"}
    assert any(f["check"] == "gitignore_pattern" and f["severity"] == "pass" for f in report["findings"])
    assert any(f["check"] == "dependency_manifest" and f["severity"] == "pass" for f in report["findings"])


def test_portability_report_flags_windows_absolute_paths(tmp_path):
    (tmp_path / ".gitignore").write_text("outputs/\nexternal/AgentHazard/\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pytest>=7\n", encoding="utf-8")
    script = tmp_path / "scripts/example.py"
    script.parent.mkdir(parents=True)
    windows_path = '"E:' + '\\\\safety_thermometer_project\\\\outputs"'
    script.write_text(f"DATA = {windows_path}\n", encoding="utf-8")

    report = build_portability_report(tmp_path)

    assert any(f["check"] == "windows_absolute_path" for f in report["findings"])


def test_portability_report_flags_tracked_large_outputs(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text("external/AgentHazard/\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pytest>=7\n", encoding="utf-8")
    output = tmp_path / "outputs/example.csv"
    output.parent.mkdir(parents=True)
    output.write_text("x\n1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", "outputs/example.csv", ".gitignore", "requirements.txt"], cwd=tmp_path, check=True, capture_output=True)

    report = build_portability_report(tmp_path)

    assert any(f["check"] == "tracked_large_output" and f["severity"] == "fail" for f in report["findings"])
