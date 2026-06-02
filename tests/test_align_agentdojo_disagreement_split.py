import json
import subprocess
import sys

import pandas as pd

from scripts.align_agentdojo_disagreement_split import (
    DISAGREEMENT_COLUMNS,
    align_agentdojo_disagreement_split,
)


SCRIPT = "scripts/align_agentdojo_disagreement_split.py"


def _rows():
    rows = []
    for episode_id, label, source_batch in [
        ("episode_train", 0, "workspace_round1"),
        ("episode_val", 1, "workspace_round2"),
        ("episode_extra", 1, "slack_recovery1"),
    ]:
        for step_id in [1, 2]:
            row = {
                "episode_id": episode_id,
                "step_id": step_id,
                "hook_type": "pre_step" if step_id == 1 else "post_step",
                "source_suite": "slack" if "slack" in source_batch else "workspace",
                "source_batch": source_batch,
                "future_risk_label": label,
                "oracle_violation": int(label and step_id == 2),
                "risk_score": 20.0 + label * 60.0,
            }
            for column in DISAGREEMENT_COLUMNS:
                row[column] = float(label) if column.startswith("max_") else int(label)
            rows.append(row)
    return rows


def _write_inputs(tmp_path):
    prefix = tmp_path / "prefix.csv"
    prior_manifest = tmp_path / "prior_manifest.json"
    pd.DataFrame(_rows()).to_csv(prefix, index=False)
    prior_manifest.write_text(
        json.dumps(
            {
                "episode_ids": {
                    "train": ["episode_train"],
                    "val": ["episode_val"],
                    "test": ["episode_missing"],
                }
            }
        ),
        encoding="utf-8",
    )
    return prefix, prior_manifest


def test_align_disagreement_split_uses_prior_assignment_and_records_unmatched(tmp_path):
    prefix, prior_manifest = _write_inputs(tmp_path)
    out_dir = tmp_path / "aligned"

    manifest = align_agentdojo_disagreement_split(prefix, prior_manifest, out_dir)

    train = pd.read_csv(out_dir / "agentdojo_train.csv")
    val = pd.read_csv(out_dir / "agentdojo_val.csv")
    test = pd.read_csv(out_dir / "agentdojo_test.csv")
    assert set(train["episode_id"]) == {"episode_train"}
    assert set(val["episode_id"]) == {"episode_val"}
    assert test.empty
    assert manifest["episode_counts"] == {"train": 1, "val": 1, "test": 0}
    assert manifest["row_counts"] == {"train": 2, "val": 2, "test": 0}
    assert manifest["unmatched_extra_episodes"] == ["episode_extra"]
    assert manifest["unmatched_extra_episode_count"] == 1
    assert manifest["missing_episodes"] == ["episode_missing"]
    assert manifest["missing_episode_count"] == 1
    assert all(column in train.columns for column in DISAGREEMENT_COLUMNS)
    assert "source_suite" in train.columns
    assert "source_batch" in train.columns


def test_align_disagreement_split_cli_writes_manifest(tmp_path):
    prefix, prior_manifest = _write_inputs(tmp_path)
    out_dir = tmp_path / "aligned"

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--input",
            str(prefix),
            "--prior-manifest",
            str(prior_manifest),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["alignment_method"] == "strict_prior_episode_assignment"
    assert (out_dir / "split_manifest.json").exists()
