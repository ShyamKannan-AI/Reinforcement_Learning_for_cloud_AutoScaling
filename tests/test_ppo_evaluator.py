import numpy as np

from src.evaluation.ppo_evaluator import PPOEvaluator


def test_ppo_evaluator_runs():

    evaluator = PPOEvaluator(
        model_path="models/ppo_vertical_scaler.pt",
        max_steps=10,
        seed=42,
    )

    results = evaluator.evaluate(
        num_episodes=1,
    )

    assert "episode_rewards" in results
    assert "records" in results

    assert len(
        results["episode_rewards"]
    ) == 1

    assert len(
        results["records"]
    ) == 1

    assert len(
        results["records"][0]
    ) == 10


def test_ppo_evaluation_record():

    evaluator = PPOEvaluator(
        model_path="models/ppo_vertical_scaler.pt",
        max_steps=5,
        seed=42,
    )

    results = evaluator.evaluate(
        num_episodes=1,
    )

    record = results[
        "records"
    ][0][0]

    expected_keys = {
        "step",
        "action",
        "reward",
        "request_rate",
        "cpu_utilization",
        "memory_utilization",
        "latency",
        "cost",
        "sla_violation",
        "replicas",
        "cpu_per_replica",
        "memory_per_replica",
    }

    assert expected_keys.issubset(
        record.keys()
    )

    assert 0 <= record["action"] <= 4
    assert record["replicas"] >= 1