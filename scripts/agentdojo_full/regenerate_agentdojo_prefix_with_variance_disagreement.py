from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regenerate_agentdojo_prefix_with_disagreement import (  # noqa: E402
    DEFAULT_SOURCE_BATCHES as BASE_SOURCE_BATCHES,
    DEFAULT_SOURCE_SUITES as BASE_SOURCE_SUITES,
    DEFAULT_TRACE_INPUTS as BASE_TRACE_INPUTS,
    DISAGREEMENT_COLUMNS as HEURISTIC_DISAGREEMENT_COLUMNS,
)
from scripts.split_prefix_dataset import split_prefix_dataset  # noqa: E402
from src.features.disagreement import (  # noqa: E402
    VARIANCE_CUMULATIVE_COLUMNS,
    VARIANCE_DISAGREEMENT_COLUMNS,
    notes_leakage_audit,
)
from src.monitor.logger import load_trace_events  # noqa: E402
from src.monitor.replay import make_prefix_dataset  # noqa: E402


DEFAULT_CANONICAL = "outputs/agentdojo_full_combined/agentdojo_full_prefix_dataset.csv"
DEFAULT_HEURISTIC = "outputs/agentdojo_full_disagreement/agentdojo_full_disagreement_prefix_dataset.csv"
DEFAULT_AUDIT = "outputs/agentdojo_expansion_outcome_audit.json"
DEFAULT_PRIOR_MANIFEST = "outputs/agentdojo_full_combined/splits/split_manifest.json"
DEFAULT_VARIANCE_DIR = "outputs/agentdojo_full_variance_disagreement"
DEFAULT_HEURISTIC_VARIANCE_DIR = "outputs/agentdojo_full_heuristic_variance_disagreement"
DEFAULT_SEED = 42
SEMANTIC_KEY = ["episode_id", "step_id", "hook_type"]
MERGE_KEY = ["source_batch", *SEMANTIC_KEY]
PREFIX_VISIBLE_TRACE_FIELDS = [
    "user_instruction",
    "plan_summary",
    "proposed_tool",
    "tool_args",
    "observation",
    "self_check",
    "notes",
]
VARIANCE_COLUMNS = [*VARIANCE_DISAGREEMENT_COLUMNS, *VARIANCE_CUMULATIVE_COLUMNS]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _trace_sources_from_audit(audit_path: str | Path) -> list[dict[str, str]]:
    audit = _load_json(audit_path)
    sources: list[dict[str, str]] = []
    for section in ["batches", "recovery_batches"]:
        for row in audit.get(section, []):
            trace_path = row.get("merged_trace")
            if not trace_path:
                continue
            sources.append(
                {
                    "trace": str(trace_path),
                    "source_suite": str(row.get("suite", "")),
                    "source_batch": str(row.get("batch", "")),
                    "source_group": str(row.get("source_group", section)),
                }
            )
    return sources


def _default_trace_sources(audit_path: str | Path) -> list[dict[str, str]]:
    sources = [
        {
            "trace": str(trace),
            "source_suite": str(suite),
            "source_batch": str(batch),
            "source_group": "base_multisuite",
        }
        for trace, suite, batch in zip(BASE_TRACE_INPUTS, BASE_SOURCE_SUITES, BASE_SOURCE_BATCHES)
    ]
    sources.extend(_trace_sources_from_audit(audit_path))
    return sources


def _feature_frame_from_trace(source: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    trace_path = Path(source["trace"])
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing saved trace: {trace_path}")
    events = load_trace_events(str(trace_path))
    notes_audit = notes_leakage_audit(events)
    df = make_prefix_dataset(str(trace_path))
    if df.empty:
        feature_df = pd.DataFrame(columns=[*MERGE_KEY, *VARIANCE_COLUMNS])
    else:
        missing = [column for column in [*SEMANTIC_KEY, *VARIANCE_COLUMNS] if column not in df.columns]
        if missing:
            raise ValueError(f"{trace_path} did not produce variance columns: {missing}")
        feature_df = df[[*SEMANTIC_KEY, *VARIANCE_COLUMNS]].copy()
        feature_df.insert(0, "source_batch", source["source_batch"])
    duplicate_rows = int(feature_df.duplicated(subset=MERGE_KEY, keep=False).sum()) if not feature_df.empty else 0
    if duplicate_rows:
        feature_df = feature_df.drop_duplicates(subset=MERGE_KEY, keep="last")
    return feature_df, {
        **source,
        "rows": int(len(feature_df)),
        "episodes": int(feature_df["episode_id"].nunique()) if "episode_id" in feature_df.columns else 0,
        "duplicate_feature_rows_dropped": duplicate_rows,
        "notes_leakage_audit": notes_audit,
    }


def _build_variance_lookup(trace_sources: list[dict[str, str]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    summaries = []
    for source in trace_sources:
        frame, summary = _feature_frame_from_trace(source)
        frames.append(frame)
        summaries.append(summary)
    if not frames:
        raise ValueError("No trace sources were provided.")
    lookup = pd.concat(frames, ignore_index=True, sort=False)
    duplicate_rows = int(lookup.duplicated(subset=MERGE_KEY, keep=False).sum())
    if duplicate_rows:
        rank = lookup["source_batch"].astype(str).str.contains("recovery", case=False, na=False).astype(int)
        lookup = (
            lookup.assign(_recovery_rank=rank, _input_order=range(len(lookup)))
            .sort_values(
                by=[*MERGE_KEY, "_recovery_rank", "_input_order"],
                ascending=[True, True, True, True, False, True],
                kind="stable",
            )
            .drop_duplicates(subset=MERGE_KEY, keep="last")
            .sort_values("_input_order", kind="stable")
            .drop(columns=["_recovery_rank", "_input_order"])
            .reset_index(drop=True)
        )
    return lookup, summaries


def _aggregate_notes_leakage_audit(trace_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    outcome_term_hits: Counter[str] = Counter()
    notes_event_count = 0
    final_notes_event_count = 0
    examples = []
    for summary in trace_summaries:
        audit = summary.get("notes_leakage_audit", {})
        notes_event_count += int(audit.get("notes_event_count", 0))
        final_notes_event_count += int(audit.get("final_notes_event_count", 0))
        outcome_term_hits.update(audit.get("outcome_term_hits", {}))
        for example in audit.get("outcome_term_event_examples", []):
            if len(examples) >= 20:
                break
            enriched = dict(example)
            enriched["trace"] = summary.get("trace")
            enriched["source_batch"] = summary.get("source_batch")
            examples.append(enriched)
    return {
        "notes_event_count": notes_event_count,
        "final_notes_event_count": final_notes_event_count,
        "outcome_term_hits": dict(sorted(outcome_term_hits.items())),
        "outcome_term_event_examples": examples,
        "final_event_notes_used_for_stream_q": False,
        "allowed_notes_hook_types_for_stream_q": ["pre_step", "post_step"],
        "interpretation": (
            "Final-hook notes are audited but excluded from stream q features; "
            "pre_step/post_step notes are allowed because they are prefix-visible."
        ),
    }


def _drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.drop(columns=[column for column in columns if column in df.columns])


def _attach_variance_features(base_df: pd.DataFrame, lookup: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    missing_base = [column for column in MERGE_KEY if column not in base_df.columns]
    if missing_base:
        raise ValueError("Canonical dataset is missing merge columns: " + ", ".join(missing_base))
    df = _drop_columns(base_df.copy(), VARIANCE_COLUMNS)
    merged = df.merge(lookup[[*MERGE_KEY, *VARIANCE_COLUMNS]], on=MERGE_KEY, how="left", validate="one_to_one")
    missing_feature_rows = int(merged[VARIANCE_COLUMNS].isna().any(axis=1).sum())
    if missing_feature_rows:
        missing_examples = merged.loc[
            merged[VARIANCE_COLUMNS].isna().any(axis=1),
            MERGE_KEY,
        ].head(20).to_dict("records")
        raise ValueError(
            "Could not align variance features for canonical rows: "
            + json.dumps(missing_examples, sort_keys=True)
        )
    return merged, missing_feature_rows


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = Counter("" if pd.isna(value) else str(value) for value in df[column].tolist())
    return dict(sorted(counts.items()))


def _split_overlap_check(manifest: dict[str, Any]) -> dict[str, Any]:
    episode_ids = {
        split: set(str(value) for value in manifest.get("episode_ids", {}).get(split, []))
        for split in ["train", "val", "test"]
    }
    overlaps = {
        "train_val": sorted(episode_ids["train"] & episode_ids["val"]),
        "train_test": sorted(episode_ids["train"] & episode_ids["test"]),
        "val_test": sorted(episode_ids["val"] & episode_ids["test"]),
    }
    return {"passed": not any(overlaps.values()), "overlaps": overlaps}


def _write_feature_manifest(
    out_dir: Path,
    prefix_path: Path,
    split_manifest: dict[str, Any],
    *,
    variant: str,
    include_heuristic: bool,
    row_count: int,
    episode_count: int,
    semantic_duplicate_rows: int,
    notes_audit: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "status": "ok",
        "variant": variant,
        "prefix_dataset": str(prefix_path),
        "split_manifest": split_manifest["outputs"]["manifest"],
        "row_count": row_count,
        "episode_count": episode_count,
        "semantic_duplicate_rows": semantic_duplicate_rows,
        "heuristic_features": HEURISTIC_DISAGREEMENT_COLUMNS if include_heuristic else [],
        "variance_features": VARIANCE_DISAGREEMENT_COLUMNS,
        "variance_cumulative_features": VARIANCE_CUMULATIVE_COLUMNS,
        "prefix_visible_fields_used_for_variance": PREFIX_VISIBLE_TRACE_FIELDS,
        "notes_leakage_audit": notes_audit,
        "forbidden_fields_not_used_for_variance": [
            "future_risk_label",
            "future_severity",
            "t_risk",
            "lead_time_if_alert_now",
            "oracle_violation",
            "utility",
            "security",
            "test split",
        ],
        "rules": {
            "validation_only_model_selection": True,
            "will_call_provider": False,
            "will_fit_calibration": False,
            "will_run_agentdojo": False,
            "will_train_risk_estimator": False,
        },
    }
    path = out_dir / "feature_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _write_variant(
    df: pd.DataFrame,
    out_dir: str | Path,
    prefix_filename: str,
    prior_manifest_path: str | Path,
    *,
    variant: str,
    include_heuristic: bool,
    seed: int,
    notes_audit: dict[str, Any],
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = out_dir / prefix_filename
    df.to_csv(prefix_path, index=False)
    semantic_duplicate_rows = int(df.duplicated(subset=SEMANTIC_KEY, keep=False).sum())
    if semantic_duplicate_rows:
        raise ValueError(f"{variant} has semantic duplicate rows.")
    split_manifest = split_prefix_dataset(
        input_path=prefix_path,
        out_dir=out_dir / "splits",
        seed=seed,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        prior_manifest_path=prior_manifest_path,
    )
    split_manifest["episode_overlap_check"] = _split_overlap_check(split_manifest)
    split_manifest["semantic_duplicate_rows"] = semantic_duplicate_rows
    split_manifest_path = Path(split_manifest["outputs"]["manifest"])
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8")
    feature_manifest = _write_feature_manifest(
        out_dir,
        prefix_path,
        split_manifest,
        variant=variant,
        include_heuristic=include_heuristic,
        row_count=int(len(df)),
        episode_count=int(df["episode_id"].nunique()),
        semantic_duplicate_rows=semantic_duplicate_rows,
        notes_audit=notes_audit,
    )
    return {
        "prefix_dataset": str(prefix_path),
        "splits": split_manifest["outputs"],
        "feature_manifest": str(out_dir / "feature_manifest.json"),
        "row_count": int(len(df)),
        "episode_count": int(df["episode_id"].nunique()),
        "semantic_duplicate_rows": semantic_duplicate_rows,
        "label_counts": split_manifest["label_counts"],
        "source_suite_counts": split_manifest["source_suite_counts"],
        "source_batch_counts": split_manifest["source_batch_counts"],
        "utility_counts": {
            split: _value_counts(pd.read_csv(path), "utility")
            for split, path in split_manifest["outputs"].items()
            if split in {"train", "val", "test"}
        },
        "feature_manifest_payload": feature_manifest,
    }


def regenerate_agentdojo_prefix_with_variance_disagreement(
    canonical_path: str | Path = DEFAULT_CANONICAL,
    heuristic_path: str | Path = DEFAULT_HEURISTIC,
    audit_path: str | Path = DEFAULT_AUDIT,
    prior_manifest_path: str | Path = DEFAULT_PRIOR_MANIFEST,
    variance_dir: str | Path = DEFAULT_VARIANCE_DIR,
    heuristic_variance_dir: str | Path = DEFAULT_HEURISTIC_VARIANCE_DIR,
    seed: int = DEFAULT_SEED,
    trace_sources: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    trace_sources = trace_sources or _default_trace_sources(audit_path)
    variance_lookup, trace_summaries = _build_variance_lookup(trace_sources)
    notes_audit = _aggregate_notes_leakage_audit(trace_summaries)
    canonical_df = pd.read_csv(canonical_path)
    heuristic_df = pd.read_csv(heuristic_path)
    variance_df, variance_missing = _attach_variance_features(canonical_df, variance_lookup)
    heuristic_variance_df, heuristic_variance_missing = _attach_variance_features(heuristic_df, variance_lookup)

    variance_df = _drop_columns(variance_df, HEURISTIC_DISAGREEMENT_COLUMNS)
    missing_heuristic = [
        column for column in HEURISTIC_DISAGREEMENT_COLUMNS if column not in heuristic_variance_df.columns
    ]
    if missing_heuristic:
        raise ValueError("Heuristic canonical dataset is missing columns: " + ", ".join(missing_heuristic))

    variance_result = _write_variant(
        variance_df,
        variance_dir,
        "agentdojo_full_variance_disagreement_prefix_dataset.csv",
        prior_manifest_path,
        variant="variance_only",
        include_heuristic=False,
        seed=seed,
        notes_audit=notes_audit,
    )
    heuristic_variance_result = _write_variant(
        heuristic_variance_df,
        heuristic_variance_dir,
        "agentdojo_full_heuristic_variance_disagreement_prefix_dataset.csv",
        prior_manifest_path,
        variant="heuristic_plus_variance",
        include_heuristic=True,
        seed=seed,
        notes_audit=notes_audit,
    )
    return {
        "status": "ok",
        "will_call_provider": False,
        "will_run_agentdojo": False,
        "will_train_risk_estimator": False,
        "will_fit_calibration": False,
        "test_split_used_for_model_selection": False,
        "inputs": {
            "canonical": str(canonical_path),
            "heuristic": str(heuristic_path),
            "audit": str(audit_path),
            "prior_manifest": str(prior_manifest_path),
        },
        "trace_sources": trace_summaries,
        "notes_leakage_audit": notes_audit,
        "variance_feature_lookup_rows": int(len(variance_lookup)),
        "variance_feature_lookup_episodes": int(variance_lookup["episode_id"].nunique()),
        "missing_variance_feature_rows": {
            "variance_only": variance_missing,
            "heuristic_plus_variance": heuristic_variance_missing,
        },
        "outputs": {
            "variance_only": variance_result,
            "heuristic_plus_variance": heuristic_variance_result,
        },
        "new_feature_columns": {
            "variance": VARIANCE_DISAGREEMENT_COLUMNS,
            "cumulative": VARIANCE_CUMULATIVE_COLUMNS,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate full AgentDojo v2 variance disagreement datasets from saved traces only."
    )
    parser.add_argument("--canonical", default=DEFAULT_CANONICAL)
    parser.add_argument("--heuristic", default=DEFAULT_HEURISTIC)
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    parser.add_argument("--reuse-manifest", default=DEFAULT_PRIOR_MANIFEST)
    parser.add_argument("--variance-dir", default=DEFAULT_VARIANCE_DIR)
    parser.add_argument("--heuristic-variance-dir", default=DEFAULT_HEURISTIC_VARIANCE_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    result = regenerate_agentdojo_prefix_with_variance_disagreement(
        canonical_path=args.canonical,
        heuristic_path=args.heuristic,
        audit_path=args.audit,
        prior_manifest_path=args.reuse_manifest,
        variance_dir=args.variance_dir,
        heuristic_variance_dir=args.heuristic_variance_dir,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
