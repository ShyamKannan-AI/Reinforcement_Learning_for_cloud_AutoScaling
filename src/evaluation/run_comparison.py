from pathlib import Path

from src.evaluation.comparison_evaluator import (
    ComparisonEvaluator,
    save_comparison_results,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "comparison_results.npz"
)


def main():

    print("\nRunning unified evaluation...")
    print("----------------------------")

    evaluator = ComparisonEvaluator(
        max_steps=500,
        num_episodes=10,
        seed=42,
    )

    results = evaluator.evaluate_all()

    metrics = evaluator.calculate_comparison_metrics(
        results
    )

    final_results = {
        "results": results,
        "metrics": metrics,
    }

    save_comparison_results(
        final_results,
        RESULTS_PATH,
    )

    print("\nComparison completed.")

    for method, method_metrics in metrics.items():

        print(
            f"\n{method.upper()}"
        )

        print(
            "Average reward:",
            method_metrics[
                "average_reward"
            ],
        )

        print(
            "Average latency:",
            method_metrics[
                "average_latency"
            ],
        )

        print(
            "P95 latency:",
            method_metrics[
                "p95_latency"
            ],
        )

        print(
            "SLA violation rate:",
            method_metrics[
                "sla_violation_rate"
            ],
        )

        print(
            "Average replicas:",
            method_metrics[
                "average_replicas"
            ],
        )

        print(
            "Scaling events:",
            method_metrics[
                "scaling_events"
            ],
        )

        print(
            "Total cost:",
            method_metrics[
                "total_cost"
            ],
        )

    print(
        "\nResults saved to:",
        RESULTS_PATH,
    )

    print(
        "File exists:",
        RESULTS_PATH.exists(),
    )


if __name__ == "__main__":
    main()