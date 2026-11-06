from pathlib import Path

import numpy as np

from src.agents.dqn.dqn_agent import DQNAgent
from src.agents.ppo.ppo_agent import PPOAgent
from src.environment.adaptive_environment import (
    AdaptiveScalingEnvironment,
)
from src.environment.cloud_simulator import CloudSimulator
from src.evaluation.metrics import calculate_evaluation_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DQN_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "dqn_horizontal_scaler.pt"
)

PPO_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_vertical_scaler.pt"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "adaptive_evaluation.npz"
)


class AdaptiveEvaluator:
    """
    Evaluates the complete adaptive auto-scaling system.

    Components:

        DQN
            Horizontal scaling

        PPO
            Vertical scaling

        ScalingController
            Adaptive decision mechanism

        CloudSimulator
            Shared environment
    """

    def __init__(
        self,
        dqn_model_path,
        ppo_model_path,
        max_steps=500,
        seed=42,
    ):
        self.max_steps = max_steps
        self.seed = seed

        # --------------------------------------------------
        # Shared simulator
        # --------------------------------------------------

        simulator = CloudSimulator(
            max_steps=max_steps,
            seed=seed,
        )

        # --------------------------------------------------
        # Adaptive environment
        # --------------------------------------------------

        self.env = AdaptiveScalingEnvironment(
            simulator=simulator,
        )

        # --------------------------------------------------
        # DQN agent
        # --------------------------------------------------

        self.dqn_agent = DQNAgent(
            state_size=self.env.get_state_size(),
            action_size=self.env.get_dqn_action_size(),
        )

        self.dqn_agent.load(
            str(dqn_model_path)
        )

        self.dqn_agent.epsilon = 0.0

        # --------------------------------------------------
        # PPO agent
        # --------------------------------------------------

        self.ppo_agent = PPOAgent(
            state_size=self.env.get_state_size(),
            action_size=self.env.get_ppo_action_size(),
        )

        self.ppo_agent.load(
            str(ppo_model_path)
        )

    def run_episode(self):
        """
        Run one complete adaptive episode.
        """

        state = self.env.reset()

        episode_reward = 0.0
        records = []

        for step in range(self.max_steps):

            next_state, reward, done, info = (
                self.env.step(
                    dqn_agent=self.dqn_agent,
                    ppo_agent=self.ppo_agent,
                )
            )

            episode_reward += reward

            records.append(
                {
                    "step": step + 1,
                    "reward": reward,
                    "scaling_mode": info[
                        "scaling_mode"
                    ],
                    "dqn_action": info[
                        "dqn_action"
                    ],
                    "ppo_action": info[
                        "ppo_action"
                    ],
                    "request_rate": info[
                        "request_rate"
                    ],
                    "cpu_utilization": info[
                        "cpu_utilization"
                    ],
                    "memory_utilization": info[
                        "memory_utilization"
                    ],
                    "latency": info[
                        "latency"
                    ],
                    "cost": info[
                        "cost"
                    ],
                    "sla_violation": info[
                        "sla_violation"
                    ],
                    "replicas": info[
                        "replicas"
                    ],
                    "cpu_per_replica": info[
                        "cpu_per_replica"
                    ],
                    "memory_per_replica": info[
                        "memory_per_replica"
                    ],
                }
            )

            state = next_state

            if done:
                break

        return {
            "episode_reward": float(
                episode_reward
            ),
            "records": records,
        }

    def evaluate(self, num_episodes=10):
        """
        Run multiple evaluation episodes.
        """

        episode_rewards = []
        all_records = []

        for episode in range(num_episodes):

            # Use a different seed for each episode.
            self.env.simulator.rng = np.random.default_rng(
                self.seed + episode
            )

            result = self.run_episode()

            episode_rewards.append(
                result["episode_reward"]
            )

            all_records.append(
                result["records"]
            )

        metrics = calculate_evaluation_metrics(
            all_records
        )

        return {
            "episode_rewards": episode_rewards,
            "records": all_records,
            "metrics": metrics,
        }


def save_evaluation_results(results, path):
    """
    Save adaptive evaluation results.
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

    print("\nStarting adaptive evaluation...")
    print(
        "DQN model:",
        DQN_MODEL_PATH,
    )
    print(
        "PPO model:",
        PPO_MODEL_PATH,
    )
    print(
        "Results:",
        RESULTS_PATH,
    )

    if not DQN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"DQN model not found: "
            f"{DQN_MODEL_PATH}"
        )

    if not PPO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"PPO model not found: "
            f"{PPO_MODEL_PATH}"
        )

    evaluator = AdaptiveEvaluator(
        dqn_model_path=DQN_MODEL_PATH,
        ppo_model_path=PPO_MODEL_PATH,
        max_steps=500,
        seed=42,
    )

    results = evaluator.evaluate(
        num_episodes=10
    )

    save_evaluation_results(
        results,
        RESULTS_PATH,
    )

    print("\nAdaptive evaluation completed.")
    print("-----------------------------")

    print(
        "Episodes:",
        len(results["episode_rewards"]),
    )

    print(
        "Average reward:",
        np.mean(
            results["episode_rewards"]
        ),
    )

    print("\nMetrics:")

    for name, value in results["metrics"].items():
        print(
            f"{name}: {value:.4f}"
        )

    print(
        "\nResults saved to:",
        RESULTS_PATH,
    )

    print(
        "File exists:",
        RESULTS_PATH.exists(),
    )

    print(
        "File size:",
        RESULTS_PATH.stat().st_size,
        "bytes",
    )


if __name__ == "__main__":
    main()