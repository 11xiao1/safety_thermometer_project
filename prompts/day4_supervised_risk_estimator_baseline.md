We now have a stable prefix-level dataset generated from toy trajectories.

Goal:
Implement a minimal supervised Risk Estimator baseline.

Method positioning:
- The trained model is called the Risk Estimator.
- Input: trajectory-prefix Evidence Stream features from outputs/toy_prefix_dataset.csv.
- Main target: future_risk_label.
- The calibrated output will later become the 0–100 Thermometer Score.
- oracle_violation is an auxiliary supervision/diagnostic column, not the main training target.
- RL is not part of this task.

Important:
Do not integrate AgentDojo yet.
Do not add deep learning frameworks.
Do not refactor the whole repository.
Use scikit-learn only.

Please inspect:
- outputs/toy_prefix_dataset.csv
- src/models/thermometer_baseline.py
- experiments/metrics.py
- scripts/make_tables.py
- tests/

Task:
1. Implement a supervised Risk Estimator baseline in src/models/thermometer_baseline.py.
2. Use simple models:
   - LogisticRegression
   - RandomForestClassifier
3. Train to predict future_risk_label, not oracle_violation.
4. Select numeric feature columns automatically, excluding label/meta columns:
   - episode_id
   - step_id
   - hook_type
   - oracle_rules
   - policy
   - future_risk_label
   - future_severity
   - t_risk
   - lead_time_if_alert_now
5. Keep oracle_violation available for analysis, but do not use it as the primary label.
6. Train/test split should be episode-level if possible. If toy data is too small, fall back to fitting and predicting on toy data, but clearly warn in output.
7. Output predictions to:
   outputs/toy_risk_estimator_predictions.csv

The output CSV should include:
- episode_id
- step_id
- future_risk_label
- oracle_violation
- risk_score_rule_based
- risk_score_logistic
- risk_score_random_forest

Risk scores should be in 0–100 scale as uncalibrated Risk Estimator scores. Calibration will be added later.

8. Add or update a script:
   scripts/train_toy_risk_estimator.py

9. Add tests:
   - the script runs successfully on toy_prefix_dataset.csv
   - output predictions contain required columns
   - predicted risk scores are between 0 and 100
   - model target is future_risk_label, not oracle_violation

Keep all existing tests passing.

After changes, report:
- modified files
- command to train
- command to test