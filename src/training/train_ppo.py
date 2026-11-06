import os
import random

import numpy as np
import torch

from src.environment.cloud_simulator import CloudSimulator
from src.environment.ppo_env import PPOVerticalScalingEnv
from src.agents.ppo.ppo_agent import PPOAgent


def set_seed(seed):
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ppo(
    num_episodes=100,
    max_steps_per_episode=500,
    rollout_length=128,
    seed=42,
):
    """
    Train PPO for vertical resource scaling.

    The PPO agent controls:
        - CPU per replica
        - Memory per replica

    Horizontal scaling is not controlled by PPO.
    """

    set_seed(seed)

    simulator = CloudSimulator(
        max_steps=max_steps_per_episode,
        seed=seed,
    )

    env = PPOVerticalScalingEnv(
        simulator=simulator
    )

    agent = PPOAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
    )

    training_history = {
        "episode_rewards": [],
        "episode_steps": [],
        "actor_losses": [],
        "critic_losses": [],
        "entropy": [],
    }

    for episode in range(
        1,
        num_episodes + 1,
    ):

        state = env.reset()

        episode_reward = 0.0
        episode_steps = 0

        episode_actor_losses = []
        episode_critic_losses = []
        episode_entropies = []

        rollout_steps = 0

        for _ in range(
            max_steps_per_episode
        ):

            episode_steps += 1
            rollout_steps += 1

            action, log_probability, value = (
                agent.select_action(
                    state,
                    training=True,
                )
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(action)

            agent.remember(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_probability=log_probability,
                value=value,
            )

            episode_reward += reward

            state = next_state

            should_update = (
                rollout_steps
                >= rollout_length
                or done
            )

            if should_update:

                if done:
                    next_value = 0.0

                else:
                    _, _, next_value = (
                        agent.select_action(
                            state,
                            training=False,
                        )
                    )

                update_result = agent.update(
                    next_value=next_value
                )

                if update_result is not None:

                    episode_actor_losses.append(
                        update_result[
                            "actor_loss"
                        ]
                    )

                    episode_critic_losses.append(
                        update_result[
                            "critic_loss"
                        ]
                    )

                    episode_entropies.append(
                        update_result[
                            "entropy"
                        ]
                    )

                rollout_steps = 0

            if done:
                break

        average_actor_loss = (
            float(
                np.mean(
                    episode_actor_losses
                )
            )
            if episode_actor_losses
            else 0.0
        )

        average_critic_loss = (
            float(
                np.mean(
                    episode_critic_losses
                )
            )
            if episode_critic_losses
            else 0.0
        )

        average_entropy = (
            float(
                np.mean(
                    episode_entropies
                )
            )
            if episode_entropies
            else 0.0
        )

        training_history[
            "episode_rewards"
        ].append(
            episode_reward
        )

        training_history[
            "episode_steps"
        ].append(
            episode_steps
        )

        training_history[
            "actor_losses"
        ].append(
            average_actor_loss
        )

        training_history[
            "critic_losses"
        ].append(
            average_critic_loss
        )

        training_history[
            "entropy"
        ].append(
            average_entropy
        )

        print(
            f"Episode "
            f"{episode:03d}/"
            f"{num_episodes} | "
            f"Reward: "
            f"{episode_reward:8.2f} | "
            f"Actor Loss: "
            f"{average_actor_loss:8.5f} | "
            f"Critic Loss: "
            f"{average_critic_loss:8.5f} | "
            f"Entropy: "
            f"{average_entropy:7.4f} | "
            f"Steps: "
            f"{episode_steps:03d} | "
            f"CPU: "
            f"{info['cpu_per_replica']:.1f} | "
            f"Memory: "
            f"{info['memory_per_replica']:.1f}"
        )

    return agent, training_history


def save_training_history(
    history,
    path,
):
    """
    Save PPO training history.
    """

    np.savez(
        path,
        episode_rewards=np.asarray(
            history["episode_rewards"],
            dtype=np.float32,
        ),
        episode_steps=np.asarray(
            history["episode_steps"],
            dtype=np.int32,
        ),
        actor_losses=np.asarray(
            history["actor_losses"],
            dtype=np.float32,
        ),
        critic_losses=np.asarray(
            history["critic_losses"],
            dtype=np.float32,
        ),
        entropy=np.asarray(
            history["entropy"],
            dtype=np.float32,
        ),
    )


if __name__ == "__main__":

    MODEL_DIR = "models"
    RESULTS_DIR = "results"

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    agent, history = train_ppo(
        num_episodes=100,
        max_steps_per_episode=500,
        rollout_length=128,
        seed=42,
    )

    model_path = os.path.join(
        MODEL_DIR,
        "ppo_vertical_scaler.pt",
    )

    history_path = os.path.join(
        RESULTS_DIR,
        "ppo_training_history.npz",
    )

    agent.save(
        model_path
    )

    save_training_history(
        history,
        history_path,
    )

    print(
        "\nPPO training completed."
    )

    print(
        f"Model saved to: {model_path}"
    )

    print(
        f"Training history saved to: "
        f"{history_path}"
    )