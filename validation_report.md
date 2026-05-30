# Prefix Dataset Validation Report

Validated file: `outputs/toy_prefix_dataset.csv`

## 1. Columns

- `episode_id`
- `step_id`
- `hook_type`
- `f_confirm_required`
- `f_confirm_seen`
- `f_state_modifying`
- `f_irreversible`
- `f_sensitive_access`
- `f_external_send`
- `f_fallback_count`
- `f_unprotected_fallback`
- `f_plan_action_gap`
- `f_self_check_gap`
- `f_repeated_failures`
- `f_risk_keyword_count`
- `cumulative_state_modifying_count`
- `cumulative_irreversible_count`
- `cumulative_sensitive_access_count`
- `cumulative_external_send_count`
- `cumulative_fallback_count`
- `confirmation_seen_so_far`
- `max_risk_score_so_far`
- `oracle_violation`
- `oracle_rules`
- `risk_score`
- `policy`
- `future_risk_label`
- `future_severity`
- `t_risk`
- `lead_time_if_alert_now`

## 2. Episode Step Counts

| episode_id | step_count |
|---|---:|
| toy_delete_with_confirmation | 2 |
| toy_delete_without_confirmation | 2 |
| toy_email_without_confirmation | 1 |
| toy_plan_action_mismatch | 1 |
| toy_safe_read | 2 |
| toy_sensitive_no_send | 1 |
| toy_unprotected_fallback | 2 |

## 3. future_risk_label Distribution

| future_risk_label | row_count |
|---:|---:|
| 0 | 5 |
| 1 | 6 |

## 4. risk_score Range

- Minimum: `0.0`
- Maximum: `100.0`

## 5. max_risk_score_so_far Monotonicity

No decreases found.

## 6. lead_time_if_alert_now Validation

No anomalies found. For rows with `t_risk`, `lead_time_if_alert_now` equals `t_risk - step_id`; for rows without `t_risk`, lead time is blank.
