import numpy as np

from src.agents.ppo.returns import (
    calculate_returns,
    calculate_gae,
    normalize_advantages,
)


def test_calculate_returns():
    rewards = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    dones = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float32,
    )

    returns = calculate_returns(
        rewards,
        dones,
        gamma=0.99,
    )

    expected = np.array(
        [
            1.0 + 0.99 * 2.0
            + (0.99 ** 2) * 3.0,

            2.0 + 0.99 * 3.0,

            3.0,
        ],
        dtype=np.float32,
    )

    assert returns.shape == (3,)

    assert np.allclose(
        returns,
        expected,
        atol=1e-5,
    )


def test_returns_reset_after_done():
    rewards = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    dones = np.array(
        [1.0, 0.0, 1.0],
        dtype=np.float32,
    )

    returns = calculate_returns(
        rewards,
        dones,
        gamma=0.99,
    )

    expected = np.array(
        [
            1.0,
            2.0 + 0.99 * 3.0,
            3.0,
        ],
        dtype=np.float32,
    )

    assert np.allclose(
        returns,
        expected,
        atol=1e-5,
    )


def test_calculate_gae():
    rewards = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    values = np.array(
        [0.5, 1.0, 1.5],
        dtype=np.float32,
    )

    dones = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float32,
    )

    advantages = calculate_gae(
        rewards,
        values,
        dones,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert advantages.shape == (3,)

    assert np.all(
        np.isfinite(advantages)
    )


def test_gae_terminal_transition():
    rewards = np.array(
        [1.0],
        dtype=np.float32,
    )

    values = np.array(
        [0.5],
        dtype=np.float32,
    )

    dones = np.array(
        [1.0],
        dtype=np.float32,
    )

    advantages = calculate_gae(
        rewards,
        values,
        dones,
        gamma=0.99,
        gae_lambda=0.95,
    )

    expected = 1.0 - 0.5

    assert np.isclose(
        advantages[0],
        expected,
        atol=1e-6,
    )


def test_normalize_advantages():
    advantages = np.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    )

    normalized = normalize_advantages(
        advantages
    )

    assert normalized.shape == advantages.shape

    assert np.isclose(
        np.mean(normalized),
        0.0,
        atol=1e-6,
    )

    assert np.isclose(
        np.std(normalized),
        1.0,
        atol=1e-6,
    )


def test_normalize_empty_advantages():
    advantages = np.array(
        [],
        dtype=np.float32,
    )

    normalized = normalize_advantages(
        advantages
    )

    assert normalized.shape == (0,)