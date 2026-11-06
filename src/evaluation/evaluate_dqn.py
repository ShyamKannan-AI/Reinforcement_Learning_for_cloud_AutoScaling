from pathlib import Path

import numpy as np

from src.environment.cloud_simulator import CloudSimulator
from src.environment.cloud_env import CloudAutoScalingEnv
from src.agents.dqn.dqn_agent import DQNAgent
from src.evaluation.evaluator import DQNEvaluator


# Project root:
# rl-cloud-autoscaling/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "dqn_horizontal_scaler.pt"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "dqn_evaluation.npz"


def save_evaluation_results(results, path):
    """
    Save DQN evaluation results to an NPZ file.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
    str(path),
    episode_rewards=np.asarray(
        results["episode_rewards"],
        dtype=np.float32,
    ),
    records=np.asarray(
        results["records"],
        dtype=object,
    ),
)


def main():

    print("\nStarting DQN evaluation...")
    print("Project root:", PROJECT_ROOT)
    print("Model path:", MODEL_PATH)
    print("Results path:", RESULTS_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"DQN model not found: {MODEL_PATH}"
        )

    simulator = CloudSimulator(
        max_steps=500,
        seed=42,
    )

    env = CloudAutoScalingEnv(
        simulator=simulator,
    )

    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
    )

    agent.load(str(MODEL_PATH))

    # Disable exploration during evaluation.
    agent.epsilon = 0.0

    evaluator = DQNEvaluator(
        env=env,
        agent=agent,
    )

    results = evaluator.evaluate(
        num_episodes=10,
        max_steps=500,
    )

    save_evaluation_results(
        results,
        RESULTS_PATH,
    )

    average_reward = np.mean(
        results["episode_rewards"]
    )

    print("\nDQN evaluation completed.")
    print("-------------------------")
    print("Episodes:", len(results["episode_rewards"]))
    print("Average reward:", average_reward)
    print("Model:", MODEL_PATH)
    print("Results saved to:", RESULTS_PATH)
    print("File exists:", RESULTS_PATH.exists())
    print("File size:", RESULTS_PATH.stat().st_size, "bytes")


if __name__ == "__main__":
    main()