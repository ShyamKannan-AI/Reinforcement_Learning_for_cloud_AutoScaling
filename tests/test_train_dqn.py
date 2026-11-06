import numpy as np

from src.training.train_dqn import train_dqn


def test_dqn_training_pipeline():

    agent, history = train_dqn(
        num_episodes=2,
        max_steps_per_episode=10,
        seed=42,
    )

    assert agent is not None

    assert len(history["episode_rewards"]) == 2
    assert len(history["episode_losses"]) == 2
    assert len(history["episode_steps"]) == 2
    assert len(history["epsilon"]) == 2

    assert np.all(
        np.isfinite(
            history["episode_rewards"]
        )
    )

    assert np.all(
        np.isfinite(
            history["episode_losses"]
        )
    )