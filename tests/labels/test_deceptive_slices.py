import pandas as pd

from src.eval.deceptive_slices import SLICE_COLUMNS, add_deceptive_slice_flags, slice_manifest


def test_deceptive_slice_builder_uses_judge_flags_and_proxy_rules():
    df = pd.DataFrame(
        [
            {
                "future_risk_label": 1,
                "deception_indicator": True,
                "surface_benign_indicator": True,
                "q_action": 0.9,
                "q_self_check": 0.1,
                "d_plan_action_gap": 0.8,
                "d_intent_action_gap": 0.7,
            },
            {
                "future_risk_label": 0,
                "deception_indicator": True,
                "surface_benign_indicator": True,
                "q_action": 0.2,
                "q_self_check": 0.1,
                "d_plan_action_gap": 0.0,
                "d_intent_action_gap": 0.0,
            },
        ]
    )

    flagged = add_deceptive_slice_flags(df)
    manifest = slice_manifest(flagged)

    assert all(column in flagged.columns for column in SLICE_COLUMNS)
    assert flagged.loc[0, "surface_benign_risky"]
    assert flagged.loc[0, "deceptive_or_miscalibrated"]
    assert flagged.loc[0, "self_check_low_external_high"]
    assert flagged.loc[0, "plan_benign_action_risky"]
    assert not flagged.loc[1, "surface_benign_risky"]
    assert manifest["claim_final_results"] is False


def test_deceptive_slice_builder_can_disable_proxy_rules():
    df = pd.DataFrame(
        [
            {
                "future_risk_label": 1,
                "deception_indicator": False,
                "surface_benign_indicator": False,
                "q_action": 0.9,
                "q_self_check": 0.1,
            }
        ]
    )

    flagged = add_deceptive_slice_flags(df, use_proxy_rules=False)

    assert not flagged.loc[0, "deceptive_or_miscalibrated"]
    assert flagged.loc[0, "self_check_low_external_high"]
