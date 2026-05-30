# Source Log

## Core manuscript

Safety Thermometer: Calibrated Early-Warning Risk Monitoring for LLM Agents.

Main implementation requirements extracted from the manuscript:

- Runtime monitor observes user instruction, plan/rationale, proposed tool,
  tool arguments, observation, state delta, and optional self-check.
- Monitor runs at pre-step, post-step, and checkpoint hooks.
- Score estimates severity-weighted future risk from current trajectory prefix.
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

## SimpRead note: SEFZ / specification violation

Useful idea: represent agent execution as an annotated execution trace, then
use deterministic reachability/path rules to check whether a dangerous action
was reached without an appropriate safety gate such as explicit confirmation.

Implementation consequence: use deterministic oracle labels when possible,
then use those labels to train a risk thermometer.

## Collaborator idea

"Extract trajectories and train a thermometer / risk monitor, potentially
with RL."

Practical interpretation:

1. Extract trajectories.
2. Convert them to prefix-level training examples.
3. Train a supervised risk model first.
4. Add RL later for intervention policy learning, not as the first step.
