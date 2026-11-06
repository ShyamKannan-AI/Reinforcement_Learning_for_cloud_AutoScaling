import numpy as np

from src.training.train_ppo import (
    train_ppo,
    save_training_history,
)


def test_train_ppo_small_run(tmp_path):
    agent, history = train_ppo(
        num_episodes=2,
        max_steps_per_episode=10,
        rollout_length=5,
        seed=42,
    )

    assert agent is not None

    assert len(
        history["episode_rewards"]
    ) == 2

    assert len(
        history["episode_steps"]
    ) == 2

    assert len(
        history["actor_losses"]
    ) == 2

    assert len(
        history["critic_losses"]
    ) == 2

    assert len(
        history["entropy"]
    ) == 2

    assert all(
        np.isfinite(value)
        for value in history[
            "episode_rewards"
        ]
    )


def test_save_training_history(tmp_path):
    history = {
        "episode_rewards": [1.0, 2.0],
        "episode_steps": [10, 10],
        "actor_losses": [0.1, 0.2],
        "critic_losses": [0.3, 0.4],
        "entropy": [1.0, 0.9],
    }

    output_path = (
        tmp_path
        / "ppo_history.npz"
    )

    save_training_history(
        history,
        output_path,
    )

    loaded = np.load(
        output_path
    )

    assert np.allclose(
        loaded["episode_rewards"],
        [1.0, 2.0],
    )

    assert np.allclose(
        loaded["episode_steps"],
        [10, 10],
    )

    assert np.allclose(
        loaded["actor_losses"],
        [0.1, 0.2],
    )

    assert np.allclose(
        loaded["critic_losses"],
        [0.3, 0.4],
    )

    assert np.allclose(
        loaded["entropy"],
        [1.0, 0.9],
    )