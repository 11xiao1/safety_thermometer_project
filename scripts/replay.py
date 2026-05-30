from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from src.monitor.replay import make_prefix_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    df = make_prefix_dataset(args.trace)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pred_path = Path(args.out).with_name("toy_predictions.csv")
    df.to_csv(pred_path, index=False)
    print(f"Wrote {args.out} and {pred_path}")


if __name__ == "__main__":
    main()
