from pathlib import Path

import numpy as np

from src.environment.cloud_simulator import CloudSimulator
from src.environment.cloud_env import CloudAutoScalingEnv
from src.environment.ppo_env import PPOVerticalScalingEnv

from src.agents.dqn.dqn_agent import DQNAgent
from src.agents.ppo.ppo_agent import PPOAgent

from src.baselines.threshold_scaler import ThresholdScaler

from src.evaluation.evaluator import DQNEvaluator
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


class ComparisonEvaluator:
    """
    Runs the currently implemented scaling approaches
    under controlled simulator conditions.

    Methods:
        - threshold
        - DQN
        - PPO
    """

    def __init__(
        self,
        max_steps=500,
        num_episodes=10,
        seed=42,
    ):
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.seed = seed

    def evaluate_dqn(self):
        """
        Evaluate the trained DQN horizontal scaler.
        """

        episode_rewards = []
        all_records = []

        for episode in range(self.num_episodes):

            simulator = CloudSimulator(
                max_steps=self.max_steps,
                seed=self.seed + episode,
            )

            env = CloudAutoScalingEnv(
                simulator=simulator,
            )

            agent = DQNAgent(
                state_size=env.get_state_size(),
                action_size=env.get_action_size(),
            )

            agent.load(
                str(DQN_MODEL_PATH)
            )

            agent.epsilon = 0.0

            evaluator = DQNEvaluator(
                env=env,
                agent=agent,
            )

            result = evaluator.run_episode(
                max_steps=self.max_steps
            )

            reward, records = result

            episode_rewards.append(
                reward
            )

            all_records.append(
                records
            )

        return {
            "episode_rewards": episode_rewards,
            "records": all_records,
        }

    def evaluate_threshold(self):
        """
        Evaluate the deterministic threshold scaler.
        """

        episode_rewards = []
        all_records = []

        for episode in range(self.num_episodes):

            simulator = CloudSimulator(
                max_steps=self.max_steps,
                seed=self.seed + episode,
            )

            simulator.reset()

            scaler = ThresholdScaler()
            scaler.reset()

            episode_reward = 0.0
            records = []

            for step in range(self.max_steps):

                current_info = {
                    "cpu_utilization": (
                        simulator.cpu_utilization
                    ),
                    "memory_utilization": (
                        simulator.memory_utilization
                    ),
                    "latency": simulator.latency,
                }

                action = scaler.select_action(
                    current_info
                )

                if action == ThresholdScaler.SCALE_UP:
                    simulator.scale_horizontal(1)

                elif action == ThresholdScaler.SCALE_DOWN:
                    simulator.scale_horizontal(-1)

                state, reward, done, info = (
                    simulator.step()
                )

                del state

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
                        "cost": info["cost"],
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

                if done:
                    break

            episode_rewards.append(
                episode_reward
            )

            all_records.append(
                records
            )

        return {
            "episode_rewards": episode_rewards,
            "records": all_records,
        }

    def evaluate_ppo(self):
        """
        Evaluate the trained PPO vertical scaler.
        """

        episode_rewards = []
        all_records = []

        for episode in range(self.num_episodes):

            simulator = CloudSimulator(
                max_steps=self.max_steps,
                seed=self.seed + episode,
            )

            env = PPOVerticalScalingEnv(
                simulator=simulator
            )

            agent = PPOAgent(
                state_size=env.get_state_size(),
                action_size=env.get_action_size(),
            )

            agent.load(
                str(PPO_MODEL_PATH)
            )

            state = env.reset()

            episode_reward = 0.0
            records = []

            for step in range(self.max_steps):

                action, _, _ = agent.select_action(
                    state,
                    training=False,
                )

                next_state, reward, done, info = (
                    env.step(action)
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
                        "cost": info["cost"],
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

            episode_rewards.append(
                episode_reward
            )

            all_records.append(
                records
            )

        return {
            "episode_rewards": episode_rewards,
            "records": all_records,
        }

    def evaluate_all(self):
        """
        Evaluate all currently implemented approaches.
        """

        return {
            "threshold": self.evaluate_threshold(),
            "dqn": self.evaluate_dqn(),
            "ppo": self.evaluate_ppo(),
        }

    def calculate_comparison_metrics(
        self,
        results,
    ):
        """
        Calculate common metrics for every method.
        """

        comparison = {}

        for method, method_results in results.items():

            comparison[method] = (
                calculate_evaluation_metrics(
                    method_results["records"]
                )
            )

        return comparison


def save_comparison_results(
    results,
    path,
):
    """
    Save numerical comparison metrics.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = results["metrics"]

    methods = list(metrics.keys())

    metric_names = list(
        metrics[methods[0]].keys()
    )

    data = {
        "method": methods,
    }

    for metric in metric_names:
        data[metric] = [
            metrics[method][metric]
            for method in methods
        ]

    np.savez(
        str(path),
        **{
            key: np.asarray(value)
            for key, value in data.items()
        },
    )