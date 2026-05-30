We need to update the project documentation to clarify the current method positioning.

Current final method:
Trajectory / Tracing
→ Evidence Streams
→ Multi-source Risk Supervision
→ Supervised Risk Estimator
→ Calibration
→ Thermometer Score
→ Runtime Monitoring / Policy
→ Optional later RL for intervention policy.

Key clarification:
We train a Risk Estimator. The Risk Estimator takes trajectory-prefix Evidence Streams as input and predicts future_risk_label. Its calibrated output becomes the 0–100 Thermometer Score used for runtime monitoring.

Oracle violations are NOT the main method. They are one auxiliary source of multi-source risk supervision, plus useful for interpretable risk slices, specification-violation tests, and diagnostics.

RL is NOT the first-stage training method. It is reserved for later learning of the intervention policy: continue / verify / alert / block.

Please update only documentation and prompts, not core code.

Files to inspect and update if needed:
- README.md
- docs/source_log.md
- docs/signal_definitions.md
- docs/decision_log.md
- prompts/day3_supervised_thermometer_baseline.md if it exists

Required wording:
Use "Risk Estimator" for the trained supervised model.
Use "Thermometer Score" for the calibrated 0–100 output.
Use "future_risk_label" as the main training target.
Describe oracle_violation as an auxiliary supervision/diagnostic signal, not as the sole label.
Describe RL as optional later intervention-policy learning, not the current training method.

Do not refactor code.
Do not modify replay.py unless strictly necessary.
Do not integrate AgentDojo.
Do not introduce deep learning frameworks.

After changes, report:
- modified files
- exact wording changes
- whether any old wording still says oracle is the main method