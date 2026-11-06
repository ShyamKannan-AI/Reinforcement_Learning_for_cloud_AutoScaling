import numpy as np

from src.evaluation.visualization import (
    plot_training_rewards,
    plot_training_loss,
    plot_epsilon,
)


HISTORY_PATH = "results/dqn_training_history.npz"


def main():

    history = np.load(HISTORY_PATH)

    rewards = history["episode_rewards"]
    losses = history["episode_losses"]
    epsilon = history["epsilon"]

    plot_training_rewards(
        rewards,
        save_path="results/dqn_reward.png",
    )

    plot_training_loss(
        losses,
        save_path="results/dqn_loss.png",
    )

    plot_epsilon(
        epsilon,
        save_path="results/dqn_epsilon.png",
    )


if __name__ == "__main__":
    main()