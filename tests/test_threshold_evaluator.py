import os

from src.evaluation.threshold_evaluator import (
    ThresholdEvaluator,
)


RESULTS_PATH = "results/threshold_evaluation.npz"


def test_threshold_evaluator_runs():

    evaluator = ThresholdEvaluator(
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


def test_threshold_evaluation_record():

    evaluator = ThresholdEvaluator(
        max_steps=5,
        seed=42,
    )

    results = evaluator.evaluate(
        num_episodes=1,
    )

    record = results["records"][0][0]

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

    assert 0 <= record["action"] <= 2

    assert record["replicas"] >= 1


def test_threshold_result_file_exists():

    if os.path.exists(RESULTS_PATH):
        assert os.path.getsize(
            RESULTS_PATH
        ) > 0