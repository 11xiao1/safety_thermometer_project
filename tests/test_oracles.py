from src.monitor.logger import load_trace_events, group_by_episode
from src.oracles.rules import evaluate_oracles


def test_toy_missing_confirmation_detected():
    events = load_trace_events("data/samples/toy_episodes.jsonl")
    grouped = group_by_episode(events)
    ep = grouped["toy_delete_without_confirmation"]
    results = evaluate_oracles(ep)
    assert any(r.rule_id == "O001" and r.violation for r in results)


def test_safe_confirmed_delete_not_flagged_by_o001():
    events = load_trace_events("data/samples/toy_episodes.jsonl")
    grouped = group_by_episode(events)
    ep = grouped["toy_delete_with_confirmation"]
    results = evaluate_oracles(ep)
    assert not any(r.rule_id == "O001" and r.violation for r in results)
