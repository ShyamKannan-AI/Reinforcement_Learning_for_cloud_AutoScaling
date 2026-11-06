from pathlib import Path

import numpy as np

from src.evaluation.threshold_evaluator import (
    ThresholdEvaluator,
    save_evaluation_results,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_PATH = (
    RESULTS_DIR
    / "threshold_evaluation.npz"
)


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluator = ThresholdEvaluator(
        max_steps=500,
        seed=42,
    )

    results = evaluator.evaluate(
        num_episodes=10,
    )

    save_evaluation_results(
        results,
        RESULTS_PATH,
    )

    average_reward = np.mean(
        results["episode_rewards"]
    )

    print("\nThreshold evaluation completed.")
    print("--------------------------------")
    print(
        "Episodes:",
        len(results["episode_rewards"]),
    )

    print(
        "Average reward:",
        average_reward,
    )

    print(
        "Results saved to:",
        RESULTS_PATH,
    )

    print(
        "File exists:",
        RESULTS_PATH.exists(),
    )


if __name__ == "__main__":
    main()