import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

def moving_average(values, window=10):
    """
    Calculate a moving average over a sequence of values.
    """

    values = np.asarray(values, dtype=np.float32)

    if len(values) < window:
        return values

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


def plot_training_rewards(
    rewards,
    save_path=None,
    window=10,
):
    """
    Plot episode rewards and their moving average.
    """

    rewards = np.asarray(rewards, dtype=np.float32)

    plt.figure(figsize=(10, 5))

    plt.plot(
        rewards,
        label="Episode Reward",
        alpha=0.5,
    )

    if len(rewards) >= window:
        smoothed = moving_average(
            rewards,
            window,
        )

        x = np.arange(
            window - 1,
            len(rewards),
        )

        plt.plot(
            x,
            smoothed,
            label=f"{window}-Episode Moving Average",
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Reward")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) or ".",
            exist_ok=True,
        )
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )


    plt.close()


def plot_training_loss(
    losses,
    save_path=None,
):
    """
    Plot average training loss per episode.
    """

    losses = np.asarray(losses, dtype=np.float32)

    plt.figure(figsize=(10, 5))

    plt.plot(
        losses,
        label="Training Loss",
    )

    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("DQN Training Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) or ".",
            exist_ok=True,
        )
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

    
    plt.close()


def plot_epsilon(
    epsilon_values,
    save_path=None,
):
    """
    Plot epsilon exploration decay.
    """

    epsilon_values = np.asarray(
        epsilon_values,
        dtype=np.float32,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        epsilon_values,
        label="Epsilon",
    )

    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("DQN Exploration Decay")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) or ".",
            exist_ok=True,
        )
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

    
    plt.close()
    
def plot_ppo_training_rewards(
    rewards,
    save_path=None,
    window=10,
):
    """
    Plot PPO episode rewards with an optional
    moving-average curve.
    """

    rewards = np.asarray(
        rewards,
        dtype=np.float32,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        rewards,
        label="Episode Reward",
        alpha=0.5,
    )

    if len(rewards) >= window:
        smoothed = moving_average(
            rewards,
            window,
        )

        x = np.arange(
            window - 1,
            len(rewards),
        )

        plt.plot(
            x,
            smoothed,
            label=f"{window}-Episode Moving Average",
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Reward")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(
            os.path.dirname(save_path)
            or ".",
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )


    plt.close()


def plot_ppo_actor_loss(
    losses,
    save_path=None,
):
    """
    Plot PPO Actor loss.
    """

    losses = np.asarray(
        losses,
        dtype=np.float32,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        losses,
        label="Actor Loss",
    )

    plt.xlabel("Episode")
    plt.ylabel("Actor Loss")
    plt.title("PPO Actor Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(
            os.path.dirname(save_path)
            or ".",
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )


    plt.close()


def plot_ppo_critic_loss(
    losses,
    save_path=None,
):
    """
    Plot PPO Critic loss.
    """

    losses = np.asarray(
        losses,
        dtype=np.float32,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        losses,
        label="Critic Loss",
    )

    plt.xlabel("Episode")
    plt.ylabel("Critic Loss")
    plt.title("PPO Critic Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(
            os.path.dirname(save_path)
            or ".",
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.close()


def plot_ppo_entropy(
    entropy_values,
    save_path=None,
):
    """
    Plot PPO policy entropy.

    Higher entropy generally indicates a more
    exploratory policy, while decreasing entropy
    can indicate that the policy is becoming
    more concentrated around particular actions.
    """

    entropy_values = np.asarray(
        entropy_values,
        dtype=np.float32,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        entropy_values,
        label="Policy Entropy",
    )

    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.title("PPO Policy Entropy")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(
            os.path.dirname(save_path)
            or ".",
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.close()