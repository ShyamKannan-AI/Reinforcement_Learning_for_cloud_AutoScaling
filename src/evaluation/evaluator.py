import numpy as np


class DQNEvaluator:
    """
    Evaluates a trained DQN horizontal-scaling agent.
    """

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run_episode(self, max_steps=None):
        """
        Run one evaluation episode without exploration.
        """

        state = self.env.reset()

        episode_reward = 0.0
        records = []

        if max_steps is None:
            max_steps = self.env.simulator.max_steps

        for _ in range(max_steps):

            action = self.agent.select_action(
                state,
                training=False,
            )

            next_state, reward, done, info = (
                self.env.step(action)
            )

            episode_reward += reward

            records.append(
                {
                    "reward": reward,
                    "action": action,
                    "request_rate": info["request_rate"],
                    "cpu_utilization": info[
                        "cpu_utilization"
                    ],
                    "memory_utilization": info[
                        "memory_utilization"
                    ],
                    "latency": info["latency"],
                    "cost": info["cost"],
                    "sla_violation": info[
                        "sla_violation"
                    ],
                    "replicas": info["replicas"],
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

        return episode_reward, records

    def evaluate(
        self,
        num_episodes=10,
        max_steps=None,
    ):
        """
        Evaluate the agent over multiple episodes.
        """

        episode_rewards = []
        all_records = []

        for _ in range(num_episodes):

            reward, records = self.run_episode(
                max_steps=max_steps
            )

            episode_rewards.append(reward)
            all_records.append(records)

        return {
            "episode_rewards": episode_rewards,
            "records": all_records,
        }