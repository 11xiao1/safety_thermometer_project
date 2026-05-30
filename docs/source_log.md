# Source Log

## Core manuscript

Safety Thermometer: Calibrated Early-Warning Risk Monitoring for LLM Agents.

Main implementation requirements extracted from the manuscript:

- Runtime monitor observes user instruction, plan/rationale, proposed tool,
  tool arguments, observation, state delta, and optional self-check.
- Monitor runs at pre-step, post-step, and checkpoint hooks.
- The Risk Estimator predicts `future_risk_label` from trajectory-prefix
  Evidence Streams.
- After calibration, the Risk Estimator output becomes the 0-100 Thermometer
  Score used for runtime monitoring.
- Evaluation should emphasize pre-risk AUROC, warning lead time, contained
  incident rate, severity-weighted calibration, and latency.

## SimpRead note: observability

Useful idea: traditional monitoring is insufficient for agents. We need:

- L1 Logging: structured events, prompts, responses, errors.
- L2 Metrics: cost, latency, success/failure, retry rate.
- L3 Tracing: full tool-call chain and decision trajectory.
- L4 Evaluation: quality, regression, safety/risk evaluation.

Implementation consequence: keep complete TraceEvent JSONL and make all
evaluation reproducible from traces.

## Current method positioning

The current method is:

```text
Trajectory / Tracing
-> Evidence Streams
-> Multi-source Risk Supervision
-> Supervised Risk Estimator
-> Calibration
-> Thermometer Score
-> Runtime Monitoring / Policy
-> Optional later RL for intervention-policy learning
```

The trained supervised model is the **Risk Estimator**. It predicts
`future_risk_label` from trajectory-prefix Evidence Streams. Its calibrated
output is the **Thermometer Score**.

`oracle_violation` is one auxiliary source of multi-source risk supervision
and a diagnostic signal. It is not the sole label and not the main method.

## SimpRead note: SEFZ / specification violation

Useful idea: represent agent execution as an annotated execution trace, then
use deterministic reachability/path rules to check whether a dangerous action
was reached without an appropriate safety gate such as explicit confirmation.

Implementation consequence: use deterministic oracle rules as auxiliary
supervision, interpretable risk slices, specification-violation tests, and
diagnostics for the Risk Estimator pipeline.

## Collaborator idea

"Extract trajectories and train a thermometer / risk monitor, potentially
with RL."

Practical interpretation:

1. Extract trajectories.
2. Convert them to prefix-level training examples.
3. Train a supervised Risk Estimator for `future_risk_label`.
4. Calibrate the Risk Estimator output into a Thermometer Score.
5. Add RL later only for intervention-policy learning, not as the current
   training method.
