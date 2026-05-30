# Signal Definitions

These signals are extracted from TraceEvent prefixes. They are deliberately
simple at first, so that toy data and AgentDojo traces can be processed before
building complex models.

| Signal | Meaning | Type | Model input |
|---|---|---:|---|
| confirmation_required | Proposed action needs explicit user confirmation | binary | f_confirm_required |
| confirmation_observed | Valid confirmation occurred before risky action | binary | f_confirm_seen |
| state_modifying_action | Proposed tool modifies external or persistent state | binary | f_state_modifying |
| irreversible_action_flag | Action is hard to undo, e.g. delete, send, publish, transfer | binary | f_irreversible |
| sensitive_resource_access | Reads or touches sensitive data/resource | binary | f_sensitive_access |
| external_send_action | Sends data to user, email, API, network, public system | binary | f_external_send |
| fallback_after_failure | Agent uses alternative path after failure | count | f_fallback_count |
| unprotected_fallback_used | Fallback bypasses original guardrail | binary | f_unprotected_fallback |
| plan_action_mismatch | Plan says one thing, tool action does another | score | f_plan_action_gap |
| self_check_gap | Self-check says low risk while independent evidence is high | score | f_self_check_gap |
| repeated_failure_count | Number of recent failed tool calls | count | f_repeated_failures |
| risk_keyword_count | Count of risk-related terms in instruction/plan/args | count | f_risk_keyword_count |

## Prefix-level labels

For every step prefix `x_<=t`, create:

- `future_risk_label`: whether a risk event occurs from step t onward.
- `future_severity`: normalized severity of that future event in [0, 1].
- `t_risk`: first risky step in the episode, or null.
- `lead_time_if_alert_now`: `t_risk - step_id` for risky episodes.
