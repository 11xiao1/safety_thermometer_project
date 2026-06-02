import json
import subprocess
import sys

import pandas as pd

from scripts.merge_prefix_datasets import merge_prefix_datasets


def _write_prefix(path, episode_id, risk_label=0, duplicate=False):
    rows = [
        {
            "episode_id": episode_id,
            "step_id": 1,
            "hook_type": "pre_step",
            "future_risk_label": risk_label,
            "risk_score": 10.0,
        },
        {
            "episode_id": episode_id,
            "step_id": 2,
            "hook_type": "post_step",
            "future_risk_label": risk_label,
            "risk_score": 20.0,
        },
    ]
    if duplicate:
        rows.append(dict(rows[-1]))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_merge_prefix_datasets_adds_source_batch_and_suite_and_preserves_rows(tmp_path):
    round1 = tmp_path / "round1.csv"
    round2 = tmp_path / "round2.csv"
    out = tmp_path / "combined.csv"
    _write_prefix(round1, "episode_a", risk_label=0)
    _write_prefix(round2, "episode_b", risk_label=1)

    result = merge_prefix_datasets(
        [round1, round2],
        ["workspace_round1", "slack_round1"],
        out,
        source_suites=["workspace", "slack"],
    )

    combined = pd.read_csv(out)
    assert result["status"] == "ok"
    assert result["output_rows"] == 4
    assert combined["source_batch"].tolist() == ["workspace_round1", "workspace_round1", "slack_round1", "slack_round1"]
    assert combined["source_suite"].tolist() == ["workspace", "workspace", "slack", "slack"]
    assert result["source_suite_counts"] == {"slack": 2, "workspace": 2}
    assert set(combined["episode_id"]) == {"episode_a", "episode_b"}
    assert "hook_type" in combined.columns
    assert "future_risk_label" in combined.columns


def test_merge_prefix_datasets_drops_exact_duplicates_only(tmp_path):
    round1 = tmp_path / "round1.csv"
    round2 = tmp_path / "round2.csv"
    out = tmp_path / "combined.csv"
    _write_prefix(round1, "episode_a", duplicate=True)
    _write_prefix(round2, "episode_a", duplicate=True)

    result = merge_prefix_datasets([round1, round2], ["round1", "round2"], out)

    combined = pd.read_csv(out)
    assert result["input_rows"] == 6
    assert result["output_rows"] == 4
    assert result["duplicate_rows_dropped"] == 2
    assert combined.groupby("source_batch").size().to_dict() == {"round1": 2, "round2": 2}


def test_merge_prefix_datasets_supports_six_inputs(tmp_path):
    inputs = []
    batches = []
    suites = []
    for index in range(6):
        path = tmp_path / f"batch_{index}.csv"
        _write_prefix(path, f"episode_{index}", risk_label=index % 2)
        inputs.append(path)
        batches.append(f"workspace_round{index}")
        suites.append("workspace")
    out = tmp_path / "combined.csv"

    result = merge_prefix_datasets(inputs, batches, out, source_suites=suites)

    assert result["status"] == "ok"
    assert result["output_rows"] == 12
    assert result["episode_count"] == 6


def test_merge_prefix_datasets_prefers_recovery_rows_for_conflicts(tmp_path):
    stopped = tmp_path / "slack_round1.csv"
    recovery = tmp_path / "slack_recovery1.csv"
    out = tmp_path / "combined.csv"
    _write_prefix(stopped, "episode_a", risk_label=0)
    _write_prefix(recovery, "episode_a", risk_label=1)

    result = merge_prefix_datasets(
        [stopped, recovery],
        ["slack_round1", "slack_recovery1"],
        out,
        source_suites=["slack", "slack"],
    )

    combined = pd.read_csv(out)
    assert result["conflict_rows_dropped"] == 2
    assert result["output_rows"] == 2
    assert set(combined["source_batch"]) == {"slack_recovery1"}
    assert set(combined["future_risk_label"]) == {1}


def test_merge_prefix_datasets_preserves_non_recovery_conflicts(tmp_path):
    round1 = tmp_path / "workspace_round1.csv"
    round2 = tmp_path / "workspace_round2.csv"
    out = tmp_path / "combined.csv"
    _write_prefix(round1, "episode_a", risk_label=0)
    _write_prefix(round2, "episode_a", risk_label=1)

    result = merge_prefix_datasets(
        [round1, round2],
        ["workspace_round1", "workspace_round2"],
        out,
        source_suites=["workspace", "workspace"],
    )

    combined = pd.read_csv(out)
    assert result["conflict_rows_dropped"] == 0
    assert result["output_rows"] == 4
    assert combined.groupby("source_batch").size().to_dict() == {
        "workspace_round1": 2,
        "workspace_round2": 2,
    }


def test_merge_prefix_datasets_cli_runs(tmp_path):
    round1 = tmp_path / "round1.csv"
    round2 = tmp_path / "round2.csv"
    out = tmp_path / "combined.csv"
    _write_prefix(round1, "episode_a")
    _write_prefix(round2, "episode_b")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/merge_prefix_datasets.py",
            "--input",
            str(round1),
            "--input",
            str(round2),
            "--source-batch",
            "round1",
            "--source-batch",
            "round2",
            "--source-suite",
            "workspace",
            "--source-suite",
            "workspace",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["output"] == str(out)
    assert out.exists()
