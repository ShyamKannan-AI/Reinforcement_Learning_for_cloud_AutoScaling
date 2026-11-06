import numpy as np

from src.evaluation.visualization import (
    moving_average,
    plot_ppo_training_rewards,
    plot_ppo_actor_loss,
    plot_ppo_critic_loss,
    plot_ppo_entropy,
)


def test_moving_average():
    values = np.array(
        [1, 2, 3, 4, 5],
        dtype=np.float32,
    )

    result = moving_average(
        values,
        window=3,
    )

    expected = np.array(
        [2, 3, 4],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_moving_average_short_sequence():
    values = np.array(
        [1, 2],
        dtype=np.float32,
    )

    result = moving_average(
        values,
        window=5,
    )

    np.testing.assert_array_equal(
        result,
        values,
    )
    
def test_plot_ppo_training_rewards(tmp_path):
    rewards = np.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    )

    output_path = (
        tmp_path
        / "ppo_reward.png"
    )

    plot_ppo_training_rewards(
        rewards,
        save_path=str(output_path),
    )

    assert output_path.exists()


def test_plot_ppo_actor_loss(tmp_path):
    losses = np.array(
        [1.0, 0.8, 0.6],
        dtype=np.float32,
    )

    output_path = (
        tmp_path
        / "ppo_actor_loss.png"
    )

    plot_ppo_actor_loss(
        losses,
        save_path=str(output_path),
    )

    assert output_path.exists()


def test_plot_ppo_critic_loss(tmp_path):
    losses = np.array(
        [2.0, 1.5, 1.0],
        dtype=np.float32,
    )

    output_path = (
        tmp_path
        / "ppo_critic_loss.png"
    )

    plot_ppo_critic_loss(
        losses,
        save_path=str(output_path),
    )

    assert output_path.exists()


def test_plot_ppo_entropy(tmp_path):
    entropy = np.array(
        [1.5, 1.3, 1.1],
        dtype=np.float32,
    )

    output_path = (
        tmp_path
        / "ppo_entropy.png"
    )

    plot_ppo_entropy(
        entropy,
        save_path=str(output_path),
    )

    assert output_path.exists()