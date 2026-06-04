from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.deceptive_slices import add_deceptive_slice_flags, slice_manifest  # noqa: E402


DEFAULT_INPUT = "outputs/agentdojo_full_variance_disagreement/splits/agentdojo_val.csv"
DEFAULT_JSON_OUT = "outputs/labels/deceptive_surface_benign_slice_preview.json"
DEFAULT_CSV_OUT = "outputs/labels/deceptive_surface_benign_slice_preview.csv"


def build_deceptive_surface_benign_slices(
    input_path: str | Path = DEFAULT_INPUT,
    json_out: str | Path = DEFAULT_JSON_OUT,
    csv_out: str | Path = DEFAULT_CSV_OUT,
    use_proxy_rules: bool = True,
) -> dict:
    input_path = Path(input_path)
    json_out = Path(json_out)
    csv_out = Path(csv_out)
    df = pd.read_csv(input_path)
    flagged = add_deceptive_slice_flags(df, use_proxy_rules=use_proxy_rules)
    manifest = slice_manifest(flagged)
    manifest.update(
        {
            "input": str(input_path),
            "outputs": {"json": str(json_out), "csv": str(csv_out)},
            "use_proxy_rules": use_proxy_rules,
            "claim_final_results": False,
            "rules": {
                "will_call_provider": False,
                "will_run_agentdojo": False,
                "will_run_agenthazard": False,
                "will_train": False,
                "will_calibrate": False,
                "test_split_used": False,
            },
        }
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(csv_out, index=False)
    json_out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deceptive/surface-benign slice preview flags.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    parser.add_argument("--no-proxy-rules", action="store_true")
    args = parser.parse_args()
    result = build_deceptive_surface_benign_slices(
        input_path=args.input,
        json_out=args.json_out,
        csv_out=args.csv_out,
        use_proxy_rules=not args.no_proxy_rules,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
