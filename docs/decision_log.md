# Decision Log

## Week 1 gate

- Can we record toy trajectories as TraceEvent JSONL?
- Can we replay traces offline?
- Can we produce trajectory-prefix Evidence Streams for training?
- Can we produce `future_risk_label` as the main Risk Estimator target?
- Can deterministic oracle rules provide auxiliary supervision and diagnostics
  for missing confirmation and unsafe fallback?
- Can an AgentDojo mini run produce the same schema?

## Week 2 gate

- Does AgentDojo provide enough steps per episode for prefix-level risk prediction?
- Can we locate `t_risk` reliably?
- Are there positive prefixes before the risky action?
- Does a supervised Risk Estimator beat rule-based score on pre-risk metrics?
- Can calibration convert Risk Estimator outputs into a stable 0-100
  Thermometer Score?
- Is RL still scoped only as optional later intervention-policy learning?
