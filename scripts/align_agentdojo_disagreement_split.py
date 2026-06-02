from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT = (
    "outputs/agentdojo_multisuite_disagreement/"
    "agentdojo_multisuite_disagreement_prefix_dataset.csv"
)
DEFAULT_PRIOR_MANIFEST = "outputs/agentdojo_multisuite_combined/splits/split_manifest.json"
DEFAULT_OUT_DIR = "outputs/agentdojo_multisuite_disagreement/splits"
SPLIT_NAMES = ["train", "val", "test"]
DISAGREEMENT_COLUMNS = [
    "f_intent_tool_mismatch",
    "f_plan_action_mismatch",
    "f_sensitive_tool_without_need",
    "f_external_send_without_request",
    "f_observation_error",
    "f_self_check_risk_mismatch",
    "f_fallback_after_error",
    "cumulative_observation_error_count",
    "cumulative_disagreement_count",
    "max_disagreement_score_so_far",
]


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = Counter("" if pd.isna(value) else str(value) for value in df[column].tolist())
    return dict(sorted(counts.items()))


def _load_episode_assignment(manifest_path: Path) -> dict[str, list[str]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_ids = payload.get("episode_ids")
    if not isinstance(episode_ids, dict):
        raise ValueError("Prior split manifest must contain an episode_ids object.")
    return {
        split_name: [str(episode_id) for episode_id in episode_ids.get(split_name, [])]
        for split_name in SPLIT_NAMES
    }


def _check_required_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in [
            "episode_id",
            "source_suite",
            "source_batch",
            "future_risk_label",
            *DISAGREEMENT_COLUMNS,
        ]
        if column not in df.columns
    ]


def align_agentdojo_disagreement_split(
    input_path: str | Path = DEFAULT_INPUT,
    prior_manifest_path: str | Path = DEFAULT_PRIOR_MANIFEST,
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    input_path = Path(input_path)
    prior_manifest_path = Path(prior_manifest_path)
    out_dir = Path(out_dir)

    df = pd.read_csv(input_path)
    missing_columns = _check_required_columns(df)
    if missing_columns:
        raise ValueError("Input disagreement prefix dataset is missing columns: " + ", ".join(missing_columns))

    assignment = _load_episode_assignment(prior_manifest_path)
    data_episode_ids = set(str(episode_id) for episode_id in df["episode_id"].dropna().unique())
    assigned_episode_ids = {
        episode_id
        for split_episodes in assignment.values()
        for episode_id in split_episodes
    }
    unmatched_extra_episodes = sorted(data_episode_ids - assigned_episode_ids)
    missing_episodes = sorted(assigned_episode_ids - data_episode_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        split_name: out_dir / f"agentdojo_{split_name}.csv"
        for split_name in SPLIT_NAMES
    }
    manifest: dict[str, Any] = {
        "input": str(input_path),
        "prior_manifest": str(prior_manifest_path),
        "alignment_method": "strict_prior_episode_assignment",
        "episode_ids": {},
        "row_counts": {},
        "episode_counts": {},
        "label_counts": {},
        "source_suite_counts": {},
        "source_batch_counts": {},
        "oracle_violation_counts": {},
        "unmatched_extra_episodes": unmatched_extra_episodes,
        "unmatched_extra_episode_count": len(unmatched_extra_episodes),
        "missing_episodes": missing_episodes,
        "missing_episode_count": len(missing_episodes),
        "outputs": {split_name: str(path) for split_name, path in output_paths.items()},
        "warnings": [],
        "method_notes": {
            "guardrail": "Strictly reuses the original no-disagreement episode assignment; no random resplit.",
            "scope": "This task only creates aligned split files and does not train or calibrate.",
            "test": "Test split is written for future reporting only; it is not read for training/calibration here.",
        },
    }
    if unmatched_extra_episodes:
        manifest["warnings"].append(
            f"{len(unmatched_extra_episodes)} disagreement episodes were not in the prior manifest and were not assigned."
        )
    if missing_episodes:
        manifest["warnings"].append(
            f"{len(missing_episodes)} prior manifest episodes were missing from disagreement data."
        )

    for split_name, split_episode_ids in assignment.items():
        available_split_episode_ids = [
            episode_id for episode_id in split_episode_ids if episode_id in data_episode_ids
        ]
        split_df = df[df["episode_id"].astype(str).isin(available_split_episode_ids)].copy()
        split_df.to_csv(output_paths[split_name], index=False)
        manifest["episode_ids"][split_name] = available_split_episode_ids
        manifest["row_counts"][split_name] = int(len(split_df))
        manifest["episode_counts"][split_name] = int(len(available_split_episode_ids))
        manifest["label_counts"][split_name] = _value_counts(split_df, "future_risk_label")
        manifest["source_suite_counts"][split_name] = _value_counts(split_df, "source_suite")
        manifest["source_batch_counts"][split_name] = _value_counts(split_df, "source_batch")
        manifest["oracle_violation_counts"][split_name] = _value_counts(split_df, "oracle_violation")

    manifest_path = out_dir / "split_manifest.json"
    manifest["outputs"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align disagreement-enhanced AgentDojo splits to the original episode assignment."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--prior-manifest", default=DEFAULT_PRIOR_MANIFEST)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    manifest = align_agentdojo_disagreement_split(
        input_path=args.input,
        prior_manifest_path=args.prior_manifest,
        out_dir=args.out_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
