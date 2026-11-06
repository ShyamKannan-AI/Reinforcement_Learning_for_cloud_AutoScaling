import os

import numpy as np

from src.environment.cloud_simulator import CloudSimulator
from src.environment.ppo_env import PPOVerticalScalingEnv
from src.agents.ppo.ppo_agent import PPOAgent


class PPOEvaluator:
    """
    Evaluates a trained PPO agent responsible for
    vertical resource scaling.

    PPO controls:
        - CPU allocation
        - Memory allocation

    PPO does NOT control:
        - Replica count
    """

    def __init__(
        self,
        model_path,
        max_steps=500,
        seed=42,
    ):
        self.model_path = model_path
        self.max_steps = max_steps
        self.seed = seed

        self.simulator = CloudSimulator(
            max_steps=max_steps,
            seed=seed,
        )

        self.env = PPOVerticalScalingEnv(
            simulator=self.simulator
        )

        self.agent = PPOAgent(
            state_size=self.env.get_state_size(),
            action_size=self.env.get_action_size(),
        )

        self.agent.load(model_path)

    def run_episode(self):
        """
        Run one deterministic evaluation episode.
        """

        state = self.env.reset()

        episode_reward = 0.0
        records = []

        for step in range(self.max_steps):

            action, log_probability, value = (
                self.agent.select_action(
                    state,
                    training=False,
                )
            )

            next_state, reward, done, info = (
                self.env.step(action)
            )

            episode_reward += reward

            records.append(
                {
                    "step": step + 1,
                    "action": action,
                    "reward": reward,

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

    def evaluate(
        self,
        num_episodes=10,
    ):
        """
        Evaluate PPO across multiple episodes.
        """

        episode_rewards = []
        all_records = []

        for _ in range(num_episodes):

            result = self.run_episode()

            episode_rewards.append(
                result["episode_reward"]
            )

            all_records.append(
                result["records"]
            )

        return {
            "episode_rewards": episode_rewards,
            "records": all_records,
        }


def save_evaluation_results(
    results,
    path,
):
    """
    Save PPO evaluation results as a NumPy file.
    """

    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    np.savez(
        path,
        episode_rewards=np.asarray(
            results["episode_rewards"],
            dtype=np.float32,
        ),
    )


if __name__ == "__main__":

    MODEL_PATH = (
        "models/ppo_vertical_scaler.pt"
    )

    RESULTS_PATH = (
        "results/ppo_evaluation.npz"
    )

    evaluator = PPOEvaluator(
        model_path=MODEL_PATH,
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

    print(
        "\nPPO evaluation completed."
    )

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

    print(
        "Results saved to:",
        RESULTS_PATH,
    )