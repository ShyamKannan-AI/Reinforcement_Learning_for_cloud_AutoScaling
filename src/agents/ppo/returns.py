import numpy as np


def calculate_returns(
    rewards,
    dones,
    gamma=0.99,
    next_value=0.0,
):
    """
    Calculate discounted returns for a rollout.

    Args:
        rewards:
            Sequence of rewards collected during the rollout.

        dones:
            Sequence indicating whether each transition ended
            the episode.

        gamma:
            Discount factor.

        next_value:
            Estimated value of the state after the final
            rollout transition.

    Returns:
        NumPy array containing the discounted return for
        every timestep.
    """

    rewards = np.asarray(
        rewards,
        dtype=np.float32,
    )

    dones = np.asarray(
        dones,
        dtype=np.float32,
    )

    returns = np.zeros_like(
        rewards,
        dtype=np.float32,
    )

    running_return = float(next_value)

    for step in reversed(range(len(rewards))):

        if dones[step]:
            running_return = 0.0

        running_return = (
            rewards[step]
            + gamma * running_return
        )

        returns[step] = running_return

    return returns


def calculate_gae(
    rewards,
    values,
    dones,
    gamma=0.99,
    gae_lambda=0.95,
    next_value=0.0,
):
    """
    Calculate Generalized Advantage Estimation (GAE).

    TD error:

        delta_t =
            r_t
            + gamma * V(s_{t+1})
            - V(s_t)

    GAE:

        A_t =
            delta_t
            + gamma * lambda * A_{t+1}

    Args:
        rewards:
            Rewards collected during the rollout.

        values:
            Critic value estimates V(s_t).

        dones:
            Episode termination indicators.

        gamma:
            Discount factor.

        gae_lambda:
            GAE smoothing parameter.

        next_value:
            Value estimate for the state after the final
            rollout transition.

    Returns:
        NumPy array containing the advantage for each timestep.
    """

    rewards = np.asarray(
        rewards,
        dtype=np.float32,
    )

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    dones = np.asarray(
        dones,
        dtype=np.float32,
    )

    advantages = np.zeros_like(
        rewards,
        dtype=np.float32,
    )

    gae = 0.0

    for step in reversed(range(len(rewards))):

        if step == len(rewards) - 1:
            next_state_value = float(next_value)
        else:
            next_state_value = values[step + 1]

        non_terminal = 1.0 - dones[step]

        delta = (
            rewards[step]
            + gamma
            * next_state_value
            * non_terminal
            - values[step]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * non_terminal
            * gae
        )

        advantages[step] = gae

    return advantages


def normalize_advantages(
    advantages,
    epsilon=1e-8,
):
    """
    Normalize advantages to approximately zero mean
    and unit variance.

    This improves numerical stability during PPO training.
    """

    advantages = np.asarray(
        advantages,
        dtype=np.float32,
    )

    if len(advantages) == 0:
        return advantages

    mean = np.mean(advantages)
    standard_deviation = np.std(advantages)

    return (
        advantages - mean
    ) / (
        standard_deviation + epsilon
    )