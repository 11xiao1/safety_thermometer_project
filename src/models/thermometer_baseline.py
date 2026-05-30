from __future__ import annotations

import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from src.features.extractor import FEATURE_COLUMNS


def train_baseline(csv_path: str) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["future_risk_label"])
    X = df[FEATURE_COLUMNS].fillna(0.0)
    y = df["future_risk_label"].astype(int)
    if y.nunique() < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "note": "Need both positive and negative labels."}
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    prob = model.predict_proba(X)[:, 1]
    return {
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    print(train_baseline(args.data))


if __name__ == "__main__":
    main()
