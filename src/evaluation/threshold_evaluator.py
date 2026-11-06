import numpy as np

from src.environment.cloud_simulator import CloudSimulator
from src.baselines.threshold_scaler import ThresholdScaler


class ThresholdEvaluator:
    """
    Evaluates the traditional threshold-based horizontal scaler.
    """

    def __init__(
        self,
        max_steps=500,
        seed=42,
    ):
        self.max_steps = max_steps
        self.seed = seed

    def run_episode(self):
        simulator = CloudSimulator(
            max_steps=self.max_steps,
            seed=self.seed,
        )

        scaler = ThresholdScaler()

        state = simulator.reset()
        del state

        scaler.reset()

        episode_reward = 0.0
        records = []

        for step in range(self.max_steps):
            # Build the current metrics before selecting an action.
            current_info = {
                "cpu_utilization": simulator.cpu_utilization,
                "memory_utilization": simulator.memory_utilization,
                "latency": simulator.latency,
            }

            action = scaler.select_action(current_info)

            if action == ThresholdScaler.SCALE_UP:
                simulator.scale_horizontal(1)

            elif action == ThresholdScaler.SCALE_DOWN:
                simulator.scale_horizontal(-1)

            next_state, reward, done, info = simulator.step()

            episode_reward += reward

            records.append(
                {
                    "step": step + 1,
                    "action": action,
                    "reward": reward,
                    "request_rate": info["request_rate"],
                    "cpu_utilization": info["cpu_utilization"],
                    "memory_utilization": info["memory_utilization"],
                    "latency": info["latency"],
                    "cost": info["cost"],
                    "sla_violation": info["sla_violation"],
                    "replicas": info["replicas"],
                    "cpu_per_replica": info["cpu_per_replica"],
                    "memory_per_replica": info["memory_per_replica"],
                }
            )

            if done:
                break

        return {
            "episode_reward": float(episode_reward),
            "records": records,
        }

    def evaluate(self, num_episodes=10):
        episode_rewards = []
        all_records = []

        for episode in range(num_episodes):
            # Use a different deterministic seed for each episode.
            self.seed = 42 + episode

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


def save_evaluation_results(results, path):
    np.savez(
        path,
        episode_rewards=np.asarray(
            results["episode_rewards"],
            dtype=np.float32,
        ),
        records=np.asarray(
            results["records"],
            dtype=object,
        ),
    )