from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from src.models.thermometer_baseline import train_risk_estimator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/toy_prefix_dataset.csv")
    parser.add_argument("--out", default="outputs/toy_risk_estimator_predictions.csv")
    args = parser.parse_args()

    result = train_risk_estimator(args.data, args.out)
    if result.warning:
        print(f"WARNING: {result.warning}")
    print(f"Wrote {args.out}")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
