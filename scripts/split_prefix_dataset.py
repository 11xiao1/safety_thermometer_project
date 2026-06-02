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
SPLIT_NAMES = ["train", "val", "test"]


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
    if n_episodes == 0:
        return {name: 0 for name in SPLIT_NAMES}, ["Input dataset contains no episodes."]
    if n_episodes < len([value for value in ratios.values() if value > 0]):
        warnings.append(
            "Dataset has fewer episodes than non-zero splits; at least one split will be empty."
        )
    if n_episodes < 5:
        warnings.append("Dataset is very small; created a best-effort episode-level split.")

    raw_counts = {name: n_episodes * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(raw_counts[name]) for name in SPLIT_NAMES}
    remaining = n_episodes - sum(counts.values())
    remainders = sorted(
        SPLIT_NAMES,
        key=lambda name: (raw_counts[name] - counts[name], ratios[name]),
        reverse=True,
    )
    for name in remainders[:remaining]:
        counts[name] += 1

    if n_episodes >= len([value for value in ratios.values() if value > 0]):
        for name in SPLIT_NAMES:
            if ratios[name] <= 0 or counts[name] > 0:
                continue
            donor = max(SPLIT_NAMES, key=lambda split: counts[split])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[name] += 1

    return counts, warnings


def _load_prior_episode_assignment(
    prior_manifest_path: str | Path | None,
    episode_ids: list[str],
) -> tuple[dict[str, list[str]] | None, list[str]]:
    if prior_manifest_path is None:
        return None, []
    prior_manifest_path = Path(prior_manifest_path)
    if not prior_manifest_path.exists():
        raise FileNotFoundError(f"Missing prior split manifest: {prior_manifest_path}")
    payload = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    prior_episode_ids = payload.get("episode_ids")
    if not isinstance(prior_episode_ids, dict):
        raise ValueError("Prior split manifest must contain an episode_ids object.")

    available = set(episode_ids)
    assigned: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    seen: set[str] = set()
    warnings = [f"Reused episode split assignment from {prior_manifest_path}."]
    for split_name in SPLIT_NAMES:
        for episode_id in prior_episode_ids.get(split_name, []):
            episode_id = str(episode_id)
            if episode_id in available and episode_id not in seen:
                assigned[split_name].append(episode_id)
                seen.add(episode_id)
    missing_from_current = sorted(
        str(episode_id)
        for split_name in SPLIT_NAMES
        for episode_id in prior_episode_ids.get(split_name, [])
        if str(episode_id) not in available
    )
    if missing_from_current:
        warnings.append(
            f"Prior split assignment referenced {len(missing_from_current)} episodes not present in current input."
        )
    return assigned, warnings


def _assign_with_prior_manifest(
    episode_ids: list[str],
    ratios: dict[str, float],
    seed: int,
    prior_manifest_path: str | Path | None,
) -> tuple[dict[str, list[str]], list[str]]:
    prior_assignment, warnings = _load_prior_episode_assignment(prior_manifest_path, episode_ids)
    if prior_assignment is None:
        rng = random.Random(seed)
        shuffled_episode_ids = episode_ids[:]
        rng.shuffle(shuffled_episode_ids)
        counts, count_warnings = _allocate_episode_counts(len(shuffled_episode_ids), ratios)
        warnings.extend(count_warnings)
        train_end = counts["train"]
        val_end = train_end + counts["val"]
        return {
            "train": shuffled_episode_ids[:train_end],
            "val": shuffled_episode_ids[train_end:val_end],
            "test": shuffled_episode_ids[val_end:],
        }, warnings

    assigned_ids = {episode_id for episodes in prior_assignment.values() for episode_id in episodes}
    remaining_episode_ids = [episode_id for episode_id in episode_ids if episode_id not in assigned_ids]
    if not remaining_episode_ids:
        return prior_assignment, warnings

    warnings.append(
        f"Assigned {len(remaining_episode_ids)} episodes absent from the prior manifest using seed {seed}."
    )
    rng = random.Random(seed)
    rng.shuffle(remaining_episode_ids)
    split_episodes = {name: episodes[:] for name, episodes in prior_assignment.items()}
    total_episode_count = len(episode_ids)
    for episode_id in remaining_episode_ids:
        target_split = min(
            SPLIT_NAMES,
            key=lambda split: (
                (len(split_episodes[split]) / total_episode_count) - ratios[split],
                len(split_episodes[split]),
            ),
        )
        split_episodes[target_split].append(episode_id)
    return split_episodes, warnings


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    if "future_risk_label" not in df.columns:
        return {}
    counts = Counter("" if pd.isna(value) else str(value) for value in df["future_risk_label"].tolist())
    return dict(sorted(counts.items()))


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = Counter("" if pd.isna(value) else str(value) for value in df[column].tolist())
    return dict(sorted(counts.items()))


def split_prefix_dataset(
    input_path: str | Path = DEFAULT_INPUT,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_RATIOS["train"],
    val_ratio: float = DEFAULT_RATIOS["val"],
    test_ratio: float = DEFAULT_RATIOS["test"],
    prior_manifest_path: str | Path | None = None,
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

    split_episodes, assignment_warnings = _assign_with_prior_manifest(
        episode_ids,
        ratios,
        seed,
        prior_manifest_path,
    )
    warnings.extend(assignment_warnings)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": out_dir / "agentdojo_train.csv",
        "val": out_dir / "agentdojo_val.csv",
        "test": out_dir / "agentdojo_test.csv",
    }

    manifest: dict[str, Any] = {
        "input": str(input_path),
        "prior_manifest": str(prior_manifest_path) if prior_manifest_path is not None else None,
        "seed": seed,
        "split_ratios": ratios,
        "episode_ids": split_episodes,
        "row_counts": {},
        "episode_counts": {},
        "label_counts": {},
        "source_suite_counts": {},
        "source_batch_counts": {},
        "oracle_violation_counts": {},
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
        manifest["source_suite_counts"][split_name] = _value_counts(split_df, "source_suite")
        manifest["source_batch_counts"][split_name] = _value_counts(split_df, "source_batch")
        manifest["oracle_violation_counts"][split_name] = _value_counts(split_df, "oracle_violation")

    for split_name in ["val", "test"]:
        label_counts = manifest["label_counts"].get(split_name, {})
        if "future_risk_label" in df.columns and manifest["row_counts"].get(split_name, 0) > 0 and len(label_counts) < 2:
            if split_name == "val":
                warnings.append("Validation split has one class; validation metrics may be limited.")
            else:
                warnings.append("Test split has one class; test metrics may be limited.")

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
    parser.add_argument("--reuse-manifest", default=None)
    args = parser.parse_args()

    manifest = split_prefix_dataset(
        input_path=args.input,
        out_dir=args.out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        prior_manifest_path=args.reuse_manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["warnings"]:
        print("Warnings:", file=sys.stderr)
        for warning in manifest["warnings"]:
            print(f"- {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
