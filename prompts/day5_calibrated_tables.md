The calibration layer is complete.

Current pipeline:
Trajectory Prefix
→ Evidence Streams
→ Supervised Risk Estimator
→ uncalibrated risk probability
→ Calibration
→ 0–100 Thermometer Score
→ threshold policy

Input:
outputs/toy_thermometer_scores.csv

Goal:
Generate calibrated Table 3 / Table 4 prototypes for the toy pipeline.

Important:
Do not modify replay.py.
Do not modify Risk Estimator training.
Do not modify calibration logic unless there is a bug.
Do not integrate AgentDojo.
Do not add RL.
Use future_risk_label as the evaluation target.
oracle_violation is diagnostic only, not the main target.

Task:
Update or extend scripts/make_tables.py so it can generate tables from:
outputs/toy_thermometer_scores.csv

Output:
outputs/table3_toy_thermometer.csv
outputs/table4_toy_thermometer.csv

Table 3 should compare:
- calibrated_logistic
- calibrated_random_forest
- thermometer_score

Metrics:
- AUROC, if valid
- AUPRC, if valid
- F1@50
- Pre-risk AUROC proxy, if available
- ECE or toy severity-weighted ECE proxy if feasible
- Mean score for safe prefixes
- Mean score for risky prefixes

Table 4 should report policy outcomes using thermometer_score and policy:
- total prefixes
- risky prefixes
- safe prefixes
- alert/block/verify/watch/continue counts
- risky prefixes alerted or blocked
- safe prefixes unnecessarily escalated
- false alert rate proxy
- contained incident proxy if available

Definitions:
- Use threshold 50 for binary alert-like evaluation unless using explicit policy.
- Treat alert/block as high-risk intervention.
- Treat verify as medium intervention.
- Treat watch/continue as non-blocking.
- Do not claim toy metrics are meaningful experimentally.
- Clearly mark these as toy pipeline sanity-check tables.

Add tests:
- table generation runs on toy_thermometer_scores.csv
- output files exist
- required table columns exist
- all metric values are finite or explicitly marked NA
- existing tests still pass

After changes, report:
- modified files
- commands to run
- output files generated
- tests added