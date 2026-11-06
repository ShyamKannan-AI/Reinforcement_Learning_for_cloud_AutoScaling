from src.evaluation.comparison_evaluator import (
    ComparisonEvaluator,
)


def test_comparison_evaluator_dqn():

    evaluator = ComparisonEvaluator(
        max_steps=5,
        num_episodes=1,
        seed=42,
    )

    result = evaluator.evaluate_dqn()

    assert "episode_rewards" in result
    assert "records" in result

    assert len(
        result["episode_rewards"]
    ) == 1

    assert len(
        result["records"]
    ) == 1

    assert len(
        result["records"][0]
    ) == 5


def test_comparison_evaluator_threshold():

    evaluator = ComparisonEvaluator(
        max_steps=5,
        num_episodes=1,
        seed=42,
    )

    result = evaluator.evaluate_threshold()

    assert len(
        result["episode_rewards"]
    ) == 1

    assert len(
        result["records"][0]
    ) == 5


def test_comparison_evaluator_ppo():

    evaluator = ComparisonEvaluator(
        max_steps=5,
        num_episodes=1,
        seed=42,
    )

    result = evaluator.evaluate_ppo()

    assert len(
        result["episode_rewards"]
    ) == 1

    assert len(
        result["records"][0]
    ) == 5


def test_comparison_metrics():

    evaluator = ComparisonEvaluator(
        max_steps=5,
        num_episodes=1,
        seed=42,
    )

    results = evaluator.evaluate_all()

    metrics = evaluator.calculate_comparison_metrics(
        results
    )

    assert "threshold" in metrics
    assert "dqn" in metrics
    assert "ppo" in metrics

    assert "average_latency" in metrics["dqn"]
    assert "p95_latency" in metrics["ppo"]
    assert "total_cost" in metrics["threshold"]