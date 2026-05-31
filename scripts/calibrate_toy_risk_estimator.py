from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from src.models.calibration import build_thermometer_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="outputs/toy_risk_estimator_predictions.csv")
    parser.add_argument("--out", default="outputs/toy_thermometer_scores.csv")
    parser.add_argument("--method", choices=["auto", "identity", "platt", "isotonic"], default="auto")
    args = parser.parse_args()

    result = build_thermometer_scores(args.pred, args.out, method=args.method)
    print(f"Wrote {args.out}")
    print(json.dumps({"calibration_methods": result.methods}, indent=2))


if __name__ == "__main__":
    main()
