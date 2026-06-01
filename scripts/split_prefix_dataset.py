from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = "outputs/agentdojo_mini_prefix_dataset.csv"
DEFAULT_OUT_DIR = "outputs/splits"
DEFAULT_SEED = 20260601
DEFAULT_RATIOS = {
    "train": 0.6,
    "val": 0.2,
    "test": 0.2,
}


def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    if any(value < 0 for value in ratios.values()):
        raise ValueError("Split ratios must be non-negative.")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")
    return {name: value / total for name, value in ratios.items()}


def _allocate_episode_counts(n_episodes: int, ratios: dict[str, float]) -> tuple[dict[str, int], list[str]]:
    warnings = []
    split_names = ["train", "val", "test"]
    if n_episodes == 0:
        return {name: 0 for name in split_names}, ["Input dataset contains no episodes."]
    if n_episodes < len([value for value in ratios.values() if value > 0]):
        warnings.append(
            "Dataset has fewer episodes than non-zero splits; at least one split will be empty."
        )
    if n_episodes < 5:
        warnings.append("Dataset is very small; created a best-effort episode-level split.")

    raw_counts = {name: n_episodes * ratios[name] for name in split_names}
    counts = {name: int(raw_counts[name]) for name in split_names}
    remaining = n_episodes - sum(counts.values())
    remainders = sorted(
        split_names,
        key=lambda name: (raw_counts[name] - counts[name], ratios[name]),
        reverse=True,
    )
    for name in remainders[:remaining]:
        counts[name] += 1

    if n_episodes >= len([value for value in ratios.values() if value > 0]):
        for name in split_names:
            if ratios[name] <= 0 or counts[name] > 0:
                continue
            donor = max(split_names, key=lambda split: counts[split])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[name] += 1

    return counts, warnings


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    if "future_risk_label" not in df.columns:
        return {}
    counts = Counter("" if pd.isna(value) else str(value) for value in df["future_risk_label"].tolist())
    return dict(sorted(counts.items()))


def split_prefix_dataset(
    input_path: str | Path = DEFAULT_INPUT,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_RATIOS["train"],
    val_ratio: float = DEFAULT_RATIOS["val"],
    test_ratio: float = DEFAULT_RATIOS["test"],
) -> dict[str, Any]:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    ratios = _normalize_ratios(train_ratio, val_ratio, test_ratio)

    df = pd.read_csv(input_path)
    if "episode_id" not in df.columns:
        raise ValueError("Input prefix dataset must contain an episode_id column.")

    warnings = []
    if "future_risk_label" not in df.columns:
        warnings.append("Input prefix dataset has no future_risk_label column; label counts are empty.")

    episode_ids = sorted(str(episode_id) for episode_id in df["episode_id"].dropna().unique())
    if df["episode_id"].isna().any():
        warnings.append("Rows with missing episode_id were found and will be excluded from all splits.")

    rng = random.Random(seed)
    shuffled_episode_ids = episode_ids[:]
    rng.shuffle(shuffled_episode_ids)

    counts, count_warnings = _allocate_episode_counts(len(shuffled_episode_ids), ratios)
    warnings.extend(count_warnings)

    train_end = counts["train"]
    val_end = train_end + counts["val"]
    split_episodes = {
        "train": shuffled_episode_ids[:train_end],
        "val": shuffled_episode_ids[train_end:val_end],
        "test": shuffled_episode_ids[val_end:],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": out_dir / "agentdojo_train.csv",
        "val": out_dir / "agentdojo_val.csv",
        "test": out_dir / "agentdojo_test.csv",
    }

    manifest: dict[str, Any] = {
        "input": str(input_path),
        "seed": seed,
        "split_ratios": ratios,
        "episode_ids": split_episodes,
        "row_counts": {},
        "episode_counts": {},
        "label_counts": {},
        "warnings": warnings,
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "method_notes": {
            "train": "Train split is reserved for future Risk Estimator training.",
            "val": "Validation split is reserved for future calibration and threshold selection.",
            "test": "Test split is reserved for final Table 3 / Table 4 reporting.",
            "guardrail": "Do not fit calibration on test data and do not randomly split prefix rows.",
        },
    }

    for split_name, episodes in split_episodes.items():
        split_df = df[df["episode_id"].astype(str).isin(episodes)].copy()
        split_df.to_csv(output_paths[split_name], index=False)
        manifest["row_counts"][split_name] = int(len(split_df))
        manifest["episode_counts"][split_name] = int(len(episodes))
        manifest["label_counts"][split_name] = _label_counts(split_df)

    manifest_path = out_dir / "split_manifest.json"
    manifest["outputs"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a prefix dataset by episode_id.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS["train"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS["val"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS["test"])
    args = parser.parse_args()

    manifest = split_prefix_dataset(
        input_path=args.input,
        out_dir=args.out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["warnings"]:
        print("Warnings:", file=sys.stderr)
        for warning in manifest["warnings"]:
            print(f"- {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
