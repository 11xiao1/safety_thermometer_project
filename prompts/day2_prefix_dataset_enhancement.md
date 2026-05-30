You are helping with a research prototype called Safety Thermometer.

Current status:
- The toy replay pipeline runs successfully.
- Oracle tests pass: 11 tests passed.
- The current pipeline converts TraceEvent JSONL into a prefix-level CSV.
- The next goal is to make this CSV a reliable training dataset for a future
  supervised Risk Estimator.

Method positioning:
- The Risk Estimator predicts `future_risk_label` from trajectory-prefix
  Evidence Streams.
- Its calibrated output becomes the 0-100 Thermometer Score.
- `oracle_violation` is auxiliary supervision/diagnostics, not the main label.
- RL is optional later intervention-policy learning, not the current training
  method.

Important:
Do not refactor the whole repository.
Do not introduce deep learning frameworks.
Do not integrate AgentDojo yet.
Focus only on prefix-level dataset quality.

Please inspect:
- src/monitor/replay.py
- src/features/extractor.py
- src/oracles/rules.py
- src/oracles/scoring.py
- tests/test_oracles.py
- data/samples/toy_episodes.jsonl

Task:
Enhance the replay output so that each row represents the trajectory prefix up to the current step.

Each CSV row should include stable columns for:
1. episode_id
2. step_id
3. hook_type
4. current-step features
5. cumulative prefix features:
   - cumulative_state_modifying_count
   - cumulative_irreversible_count
   - cumulative_sensitive_access_count
   - cumulative_external_send_count
   - cumulative_fallback_count
   - confirmation_seen_so_far
   - max_risk_score_so_far
6. oracle_violation
7. oracle_rules
8. risk_score
9. policy
10. future_risk_label
11. future_severity
12. t_risk
13. lead_time_if_alert_now

Definitions:
- future_risk_label = 1 if the suffix from current step onward contains a risky step or oracle violation; otherwise 0.
- future_severity should be the maximum future severity from current step onward.
- t_risk should be the first risky step in the episode, if any.
- lead_time_if_alert_now = t_risk - step_id if t_risk exists; otherwise blank or null.
- confirmation_seen_so_far should become true once an explicit confirmation has appeared earlier in the prefix.
- max_risk_score_so_far should be the maximum risk_score up to the current step.

Add tests in a new file:
- tests/test_prefix_dataset.py

Tests should verify:
1. The replay CSV contains all required columns.
2. cumulative_fallback_count increases after fallback.
3. confirmation_seen_so_far remains false for implicit confirmation.
4. max_risk_score_so_far is monotonic non-decreasing within each episode.
5. lead_time_if_alert_now is positive before the risky step, zero at the risky step, and negative after it.
6. safe episodes have future_risk_label = 0.

Keep all existing tests passing.

After changes, report:
- modified files
- new columns added
- tests added
- exact commands to run
