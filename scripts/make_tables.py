from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import pandas as pd
from experiments.metrics import early_warning_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.pred)
    metrics = early_warning_metrics(df)
    table3 = pd.DataFrame([{"Method": "Rule-based Thermometer", **metrics}])
    table3.to_csv(outdir / "table3_toy.csv", index=False)

    table4 = pd.DataFrame([{
        "Setting": "Toy",
        "Contained": metrics.get("Contained Incident Rate"),
        "Uncontained": None,
        "False alert": metrics.get("False Alert Rate"),
        "Latency": "not_measured",
    }])
    table4.to_csv(outdir / "table4_toy.csv", index=False)
    print(table3)


if __name__ == "__main__":
    main()
