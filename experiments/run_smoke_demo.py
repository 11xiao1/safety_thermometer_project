from pathlib import Path
from src.monitor.replay import make_prefix_dataset


if __name__ == "__main__":
    out = Path("outputs/toy_prefix_dataset.csv")
    out.parent.mkdir(exist_ok=True)
    df = make_prefix_dataset("data/samples/toy_episodes.jsonl")
    df.to_csv(out, index=False)
    print(df.head())
    print(f"Wrote {out}")
