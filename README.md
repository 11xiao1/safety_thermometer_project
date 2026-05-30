# Safety Thermometer Project

This repository is a minimal scaffold for implementing **Safety Thermometer**:
a calibrated early-warning risk monitor for tool-using LLM agents.

The first milestone is not a full benchmark result. The first milestone is a
working loop:

```text
toy / AgentDojo trajectory
-> TraceEvent JSONL
-> trajectory-prefix Evidence Streams
-> multi-source risk supervision
-> prefix-level dataset
-> supervised Risk Estimator
-> calibration
-> Thermometer Score
-> runtime monitoring / policy
-> Table 3 / Table 4 prototypes
```

## Method positioning

The trained supervised model is the **Risk Estimator**. It takes
trajectory-prefix Evidence Streams as input and predicts `future_risk_label`.
After calibration, the Risk Estimator output becomes the 0-100
**Thermometer Score** used for runtime monitoring.

`oracle_violation` is an auxiliary supervision and diagnostic signal, not the
sole label or main method. Oracle rules remain useful for interpretable risk
slices, specification-violation tests, and debugging the trace pipeline.

RL is not the current training method. It is reserved for optional later
intervention-policy learning, such as choosing among `continue`, `verify`,
`alert`, and `block`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/replay.py --trace data/samples/toy_episodes.jsonl --out outputs/toy_prefix_dataset.csv
python scripts/make_tables.py --pred outputs/toy_predictions.csv --outdir outputs
pytest -q
```

## Recommended first-week workflow

1. Validate toy trajectories.
2. Implement/extend `TraceEvent` and JSONL logger.
3. Generate prefix-level Evidence Streams and labels.
4. Keep deterministic oracle rules as auxiliary supervision and diagnostics.
5. Train a supervised Risk Estimator for `future_risk_label`.
6. Calibrate the Risk Estimator output into a Thermometer Score.
7. Only then add an AgentDojo adapter.

## Design rule

Keep the project **trace-first**. Every Risk Estimator, metric, calibration
step, or optional later RL intervention policy must be reproducible from saved
trajectories, not only from live agent runs.
