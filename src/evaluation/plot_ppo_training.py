import numpy as np

from src.evaluation.visualization import (
    plot_ppo_training_rewards,
    plot_ppo_actor_loss,
    plot_ppo_critic_loss,
    plot_ppo_entropy,
)


HISTORY_PATH = (
    "results/ppo_training_history.npz"
)


def main():
    history = np.load(
        HISTORY_PATH
    )

    rewards = history[
        "episode_rewards"
    ]

    actor_losses = history[
        "actor_losses"
    ]

    critic_losses = history[
        "critic_losses"
    ]

    entropy = history[
        "entropy"
    ]

    plot_ppo_training_rewards(
        rewards,
        save_path=(
            "results/ppo_reward.png"
        ),
    )

    plot_ppo_actor_loss(
        actor_losses,
        save_path=(
            "results/ppo_actor_loss.png"
        ),
    )

    plot_ppo_critic_loss(
        critic_losses,
        save_path=(
            "results/ppo_critic_loss.png"
        ),
    )

    plot_ppo_entropy(
        entropy,
        save_path=(
            "results/ppo_entropy.png"
        ),
    )


if __name__ == "__main__":
    main()