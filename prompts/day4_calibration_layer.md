Codex task:

The toy Risk Estimator pipeline is complete.

Current method:

Trajectory Prefix
→ Evidence Streams
→ Supervised Risk Estimator
→ uncalibrated risk probability
→ Calibration
→ 0–100 Thermometer Score
→ threshold policy
→ Table 3 / Table 4

Important method positioning:

- The trained model is the Risk Estimator.
- The calibrated output is the Thermometer Score.
- future_risk_label is the main calibration/evaluation target.
- oracle_violation is auxiliary supervision/diagnostic information, not the calibration target.
- RL is not part of this task.
- Do not integrate AgentDojo yet.
- Do not modify replay.py unless strictly necessary.
- Do not refactor unrelated code.
- Use scikit-learn only.

Input:

outputs/toy_risk_estimator_predictions.csv

Expected existing columns may include:

- episode_id
- step_id
- future_risk_label
- oracle_violation
- risk_score_rule_based
- risk_score_logistic
- risk_score_random_forest

Task:

1. Create:

src/models/calibration.py

1. Implement calibration utilities:

- score normalization to 0–1 probability scale
- score conversion back to 0–100 scale
- identity calibration fallback for tiny toy data
- Platt scaling if label diversity and sample count permit
- isotonic calibration if label diversity and sample count permit
- policy mapping from calibrated score:
  - 0–20 continue
  - 21–40 watch
  - 41–60 verify
  - 61–80 alert
  - 81–100 block

1. Create:

scripts/calibrate_toy_risk_estimator.py

CLI:

python scripts/calibrate_toy_risk_estimator.py --pred outputs/toy_risk_estimator_predictions.csv --out outputs/toy_thermometer_scores.csv

1. Output CSV:

outputs/toy_thermometer_scores.csv

Required columns:

- episode_id
- step_id
- future_risk_label
- oracle_violation
- raw_score_logistic
- raw_score_random_forest
- calibrated_score_logistic
- calibrated_score_random_forest
- thermometer_score
- policy

Definitions:

- raw_score_logistic should come from risk_score_logistic.
- raw_score_random_forest should come from risk_score_random_forest.
- calibrated scores must be in [0, 100].
- thermometer_score can initially be the calibrated logistic score by default, or an average/selected calibrated score if clearly documented.
- If calibration cannot be fitted due to tiny toy data, fall back to identity calibration and print a clear warning. This is acceptable for toy pipeline validation.

1. Add tests:

Create or update tests/test_calibration.py.

Tests should verify:

- calibration script runs on toy_risk_estimator_predictions.csv
- output file contains all required columns
- calibrated scores are always in [0, 100]
- policy values are only continue/watch/verify/alert/block
- thermometer_score exists and is in [0, 100]
- identity fallback works when sample count is too small or labels lack diversity
- existing tests still pass

1. Do not claim toy calibration is meaningful experimentally.

Toy calibration is only a pipeline sanity check. Real calibration requires validation data from real benchmark traces later.

After changes, report:

- modified files
- new script command
- output columns
- tests added
- exact commands to run