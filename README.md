# Safety Thermometer Project

This repository is a minimal scaffold for implementing **Safety Thermometer**:
a calibrated early-warning risk monitor for tool-using LLM agents.

The first milestone is not a full benchmark result. The first milestone is a
working loop:

```text
toy / AgentDojo trajectory
-> TraceEvent JSONL
-> signal extraction
-> deterministic oracle labels
-> prefix-level dataset
-> rule-based thermometer score
-> Table 3 / Table 4 prototypes
```

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
3. Implement deterministic oracle rules for missing confirmation and unsafe fallback.
4. Generate prefix-level data.
5. Produce toy Table 3 / Table 4 prototypes.
6. Only then add an AgentDojo adapter.

## Design rule

Keep the project **trace-first**. Every model, metric, or RL policy must be
reproducible from saved trajectories, not only from live agent runs.
