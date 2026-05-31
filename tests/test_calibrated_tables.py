import math
import subprocess
import sys

import pandas as pd


THERMOMETER_SCORES = "outputs/toy_thermometer_scores.csv"
TABLE3_COLUMNS = [
    "Method",
    "Setting",
    "AUROC",
    "AUPRC",
    "F1@50",
    "Pre-risk AUROC Proxy",
    "ECE Proxy",
    "Mean Score Safe Prefixes",
    "Mean Score Risky Prefixes",
]
TABLE4_COLUMNS = [
    "Setting",
    "Total Prefixes",
    "Risky Prefixes",
    "Safe Prefixes",
    "Alert Count",
    "Block Count",
    "Verify Count",
    "Watch Count",
    "Continue Count",
    "Risky Prefixes Alerted Or Blocked",
    "Safe Prefixes Unnecessarily Escalated",
    "False Alert Rate Proxy",
    "Contained Incident Proxy",
]


def _generate_tables(outdir):
    subprocess.run(
        [
            sys.executable,
            "scripts/make_tables.py",
            "--pred",
            THERMOMETER_SCORES,
            "--outdir",
            str(outdir),
        ],
        check=True,
    )


def _finite_or_na(value) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str):
        return value == "NA"
    return math.isfinite(float(value))


def test_thermometer_table_generation_runs(tmp_path):
    _generate_tables(tmp_path)

    assert (tmp_path / "table3_toy_thermometer.csv").exists()
    assert (tmp_path / "table4_toy_thermometer.csv").exists()


def test_thermometer_tables_have_required_columns(tmp_path):
    _generate_tables(tmp_path)

    table3 = pd.read_csv(tmp_path / "table3_toy_thermometer.csv")
    table4 = pd.read_csv(tmp_path / "table4_toy_thermometer.csv")
    assert all(column in table3.columns for column in TABLE3_COLUMNS)
    assert all(column in table4.columns for column in TABLE4_COLUMNS)


def test_thermometer_table_metrics_are_finite_or_na(tmp_path):
    _generate_tables(tmp_path)

    table3 = pd.read_csv(tmp_path / "table3_toy_thermometer.csv")
    table4 = pd.read_csv(tmp_path / "table4_toy_thermometer.csv")
    metric_columns = [column for column in table3.columns if column not in ["Method", "Setting"]]
    metric_columns += [column for column in table4.columns if column != "Setting"]

    values = table3[[column for column in table3.columns if column not in ["Method", "Setting"]]].to_numpy().ravel()
    values = list(values) + table4[[column for column in table4.columns if column != "Setting"]].to_numpy().ravel().tolist()
    assert all(_finite_or_na(value) for value in values)
