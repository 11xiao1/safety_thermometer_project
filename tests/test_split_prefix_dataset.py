import json
import subprocess
import sys

import pandas as pd

from scripts.split_prefix_dataset import split_prefix_dataset


def _write_mock_prefix_dataset(path, episode_count=5):
    rows = []
    for episode_index in range(episode_count):
        episode_id = f"episode_{episode_index}"
        for step_id in [1, 2]:
            rows.append(
                {
                    "episode_id": episode_id,
                    "step_id": step_id,
                    "hook_type": "pre_step" if step_id == 1 else "post_step",
                    "future_risk_label": episode_index % 2,
                    "risk_score": 10 + episode_index,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _read_episode_ids(path):
    df = pd.read_csv(path)
    return set(df["episode_id"].unique())


def test_split_prefix_dataset_creates_outputs_and_manifest(tmp_path):
    input_path = tmp_path / "prefix.csv"
    out_dir = tmp_path / "splits"
    _write_mock_prefix_dataset(input_path, episode_count=5)

    manifest = split_prefix_dataset(input_path=input_path, out_dir=out_dir, seed=7)

    assert (out_dir / "agentdojo_train.csv").exists()
    assert (out_dir / "agentdojo_val.csv").exists()
    assert (out_dir / "agentdojo_test.csv").exists()
    assert (out_dir / "split_manifest.json").exists()
    assert manifest["seed"] == 7
    assert manifest["episode_counts"] == {"train": 3, "val": 1, "test": 1}
    assert manifest["row_counts"] == {"train": 6, "val": 2, "test": 2}
    assert manifest["label_counts"]["train"]

    disk_manifest = json.loads((out_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert disk_manifest["episode_ids"] == manifest["episode_ids"]


def test_no_episode_id_appears_in_more_than_one_split(tmp_path):
    input_path = tmp_path / "prefix.csv"
    out_dir = tmp_path / "splits"
    _write_mock_prefix_dataset(input_path, episode_count=10)

    split_prefix_dataset(input_path=input_path, out_dir=out_dir, seed=11)

    train_episodes = _read_episode_ids(out_dir / "agentdojo_train.csv")
    val_episodes = _read_episode_ids(out_dir / "agentdojo_val.csv")
    test_episodes = _read_episode_ids(out_dir / "agentdojo_test.csv")

    assert train_episodes.isdisjoint(val_episodes)
    assert train_episodes.isdisjoint(test_episodes)
    assert val_episodes.isdisjoint(test_episodes)
    assert len(train_episodes | val_episodes | test_episodes) == 10


def test_all_prefix_rows_from_an_episode_stay_together(tmp_path):
    input_path = tmp_path / "prefix.csv"
    out_dir = tmp_path / "splits"
    _write_mock_prefix_dataset(input_path, episode_count=6)

    split_prefix_dataset(input_path=input_path, out_dir=out_dir, seed=13)

    source_df = pd.read_csv(input_path)
    for split_name in ["train", "val", "test"]:
        split_df = pd.read_csv(out_dir / f"agentdojo_{split_name}.csv")
        for episode_id in split_df["episode_id"].unique():
            assert len(split_df[split_df["episode_id"] == episode_id]) == len(
                source_df[source_df["episode_id"] == episode_id]
            )


def test_too_small_dataset_warns_and_still_writes_best_effort_split(tmp_path):
    input_path = tmp_path / "prefix.csv"
    out_dir = tmp_path / "splits"
    _write_mock_prefix_dataset(input_path, episode_count=2)

    manifest = split_prefix_dataset(input_path=input_path, out_dir=out_dir, seed=17)

    assert "Dataset has fewer episodes than non-zero splits; at least one split will be empty." in manifest["warnings"]
    assert "Dataset is very small; created a best-effort episode-level split." in manifest["warnings"]
    assert sum(manifest["episode_counts"].values()) == 2
    assert sum(manifest["row_counts"].values()) == 4


def test_splitter_cli_runs_on_mock_prefix_dataset(tmp_path):
    input_path = tmp_path / "prefix.csv"
    out_dir = tmp_path / "splits"
    _write_mock_prefix_dataset(input_path, episode_count=5)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/split_prefix_dataset.py",
            "--input",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            "19",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["seed"] == 19
    assert (out_dir / "split_manifest.json").exists()
