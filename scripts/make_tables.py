from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


RISK_ESTIMATOR_SCORE_COLUMNS = {
    "rule_based": "risk_score_rule_based",
    "logistic": "risk_score_logistic",
    "random_forest": "risk_score_random_forest",
}


def _safe_metric(fn, y_true: pd.Series, y_score: pd.Series) -> float:
    try:
        if y_true.nunique() < 2:
            return float("nan")
        return float(fn(y_true, y_score))
    except Exception:
        return float("nan")


def _method_metrics(df: pd.DataFrame, score_column: str, threshold: float = 50.0) -> dict[str, float]:
    y = df["future_risk_label"].astype(int)
    scores = df[score_column].astype(float)
    pred = (scores >= threshold).astype(int)
    risky_episode = df.groupby("episode_id")["future_risk_label"].max().astype(int)
    alert_episode = pred.groupby(df["episode_id"]).max().astype(int)
    risky_episode_ids = risky_episode[risky_episode == 1].index

    if len(risky_episode_ids):
        risky_detected = float(alert_episode.reindex(risky_episode_ids, fill_value=0).mean())
    else:
        risky_detected = float("nan")

    return {
        "AUROC": _safe_metric(roc_auc_score, y, scores),
        "AUPRC": _safe_metric(average_precision_score, y, scores),
        "F1@50": float(f1_score(y, pred, zero_division=0)),
        "False Alert Rate": float(((pred == 1) & (y == 0)).mean()),
        "Risky Detected Rate Proxy": risky_detected,
    }


def _make_risk_estimator_tables(df: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    table4_rows = []
    for method, score_column in RISK_ESTIMATOR_SCORE_COLUMNS.items():
        metrics = _method_metrics(df, score_column)
        rows.append({"Method": method, **metrics})
        table4_rows.append({
            "Method": method,
            "Setting": "Toy",
            "Threshold": 50,
            "Risky detected proxy": metrics["Risky Detected Rate Proxy"],
            "False alert rate": metrics["False Alert Rate"],
            "Latency": "not_measured",
        })

    table3 = pd.DataFrame(rows)
    table4 = pd.DataFrame(table4_rows)
    table3.to_csv(outdir / "table3_toy_risk_estimator.csv", index=False)
    table4.to_csv(outdir / "table4_toy_risk_estimator.csv", index=False)
    return table3, table4


def _make_rule_based_tables(df: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    if "risk_score_rule_based" in working.columns and "risk_score" not in working.columns:
        working["risk_score"] = working["risk_score_rule_based"]
    metrics = _method_metrics(working.rename(columns={"risk_score": "risk_score_rule_based"}), "risk_score_rule_based")
    table3 = pd.DataFrame([{"Method": "rule_based", **metrics}])
    table3.to_csv(outdir / "table3_toy.csv", index=False)

    table4 = pd.DataFrame([{
        "Method": "rule_based",
        "Setting": "Toy",
        "Threshold": 50,
        "Risky detected proxy": metrics["Risky Detected Rate Proxy"],
        "False alert rate": metrics["False Alert Rate"],
        "Latency": "not_measured",
    }])
    table4.to_csv(outdir / "table4_toy.csv", index=False)
    return table3, table4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.pred)

    if all(column in df.columns for column in RISK_ESTIMATOR_SCORE_COLUMNS.values()):
        table3, _ = _make_risk_estimator_tables(df, outdir)
    else:
        table3, _ = _make_rule_based_tables(df, outdir)
    print(table3)


if __name__ == "__main__":
    main()
