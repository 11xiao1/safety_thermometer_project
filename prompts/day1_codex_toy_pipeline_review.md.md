You are helping with a research prototype called Safety Thermometer.

Project goal:
Build a trajectory-level risk monitoring pipeline for LLM agents. The current toy pipeline should:
1. Load TraceEvent JSONL files.
2. Convert each agent step into a prefix-level sample.
3. Extract intermediate risk signals.
4. Apply deterministic oracle rules.
5. Produce a rule-based thermometer risk score.
6. Output a CSV that can later be used to train a supervised risk monitor.

Important:
Do not refactor the whole repository.
Do not introduce deep learning frameworks.
Do not modify the project goal.
Focus only on making the toy pipeline reliable.

Please inspect these files:
- README.md
- docs/source_log.md
- docs/signal_definitions.md
- docs/confirmation_state_spec.md
- src/monitor/schema.py
- src/monitor/replay.py
- src/features/extractor.py
- src/oracles/rules.py
- src/oracles/scoring.py
- tests/test_oracles.py
- data/samples/toy_episodes.jsonl

Task:
1. Review whether the current toy pipeline is logically consistent.
2. Add or improve tests for the following cases:
   - O001: a dangerous delete action without explicit confirmation should trigger a violation.
   - O005: fallback from a protected path to an unprotected path should trigger a violation.
   - O007: implicit confirmation should not count as explicit confirmation.
   - A safe read-only episode should not trigger any oracle violation.
3. Add a test that verifies replay produces a prefix-level CSV with stable required columns:
   - episode_id
   - step_id
   - hook_type
   - risk_score
   - policy
   - oracle_violation
   - oracle_rules
   - future_risk_label
   - future_severity
4. Keep all existing tests passing.

After changes, tell me:
- what files were modified
- what tests were added
- how to run the tests