import os
import random

import numpy as np
import torch

from src.environment.cloud_simulator import CloudSimulator
from src.environment.cloud_env import CloudAutoScalingEnv
from src.agents.dqn.dqn_agent import DQNAgent


def set_seed(seed):
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_dqn(
    num_episodes=100,
    max_steps_per_episode=500,
    seed=42,
):
    """
    Train the DQN agent for horizontal autoscaling.

    Returns:
        agent
        training_history
    """

    set_seed(seed)

    simulator = CloudSimulator(
        max_steps=max_steps_per_episode,
        seed=seed,
    )

    env = CloudAutoScalingEnv(
        simulator=simulator
    )

    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
    )

    training_history = {
        "episode_rewards": [],
        "episode_losses": [],
        "episode_steps": [],
        "epsilon": [],
    }

    for episode in range(1, num_episodes + 1):

        state = env.reset()

        episode_reward = 0.0
        episode_losses = []
        step_count = 0

        for step in range(max_steps_per_episode):

            step_count += 1

            # Select action.
            action = agent.select_action(
                state,
                training=True,
            )

            # Execute action.
            next_state, reward, done, info = env.step(
                action
            )

            # Store experience.
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            # Train DQN.
            loss = agent.train_step()

            if loss is not None:
                episode_losses.append(loss)

            episode_reward += reward

            state = next_state

            if done:
                break

        average_loss = (
            float(np.mean(episode_losses))
            if episode_losses
            else 0.0
        )

        training_history["episode_rewards"].append(
            episode_reward
        )

        training_history["episode_losses"].append(
            average_loss
        )

        training_history["episode_steps"].append(
            step_count
        )

        training_history["epsilon"].append(
            agent.epsilon
        )

        print(
            f"Episode {episode:03d}/{num_episodes} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Loss: {average_loss:.5f} | "
            f"Steps: {step_count:03d} | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"Replicas: {info['replicas']}"
        )

    return agent, training_history


def save_training_history(history, path):
    """
    Save training history as a NumPy file.
    """

    np.savez(
        path,
        episode_rewards=np.asarray(
            history["episode_rewards"],
            dtype=np.float32,
        ),
        episode_losses=np.asarray(
            history["episode_losses"],
            dtype=np.float32,
        ),
        episode_steps=np.asarray(
            history["episode_steps"],
            dtype=np.int32,
        ),
        epsilon=np.asarray(
            history["epsilon"],
            dtype=np.float32,
        ),
    )


if __name__ == "__main__":

    MODEL_DIR = "models"
    RESULTS_DIR = "results"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    agent, history = train_dqn(
        num_episodes=100,
        max_steps_per_episode=500,
        seed=42,
    )

    model_path = os.path.join(
        MODEL_DIR,
        "dqn_horizontal_scaler.pt",
    )

    history_path = os.path.join(
        RESULTS_DIR,
        "dqn_training_history.npz",
    )

    agent.save(model_path)

    save_training_history(
        history,
        history_path,
    )

    print("\nTraining completed.")
    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")