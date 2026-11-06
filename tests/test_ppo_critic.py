import numpy as np
import torch

from src.agents.ppo.critic import PPOCritic


def create_state():
    return torch.tensor(
        np.array(
            [
                0.5,
                0.6,
                0.4,
                0.2,
                0.15,
                0.25,
                0.125,
                0.05,
            ],
            dtype=np.float32,
        )
    )


def test_critic_output_shape():
    critic = PPOCritic()

    state = create_state()

    value = critic(state)

    assert value.shape == (1,)


def test_critic_batch_output_shape():
    critic = PPOCritic()

    states = torch.randn(16, 8)

    values = critic(states)

    assert values.shape == (16, 1)


def test_critic_output_is_finite():
    critic = PPOCritic()

    states = torch.randn(10, 8)

    values = critic(states)

    assert torch.all(torch.isfinite(values))


def test_critic_accepts_different_states():
    critic = PPOCritic()

    state_1 = torch.randn(8)
    state_2 = torch.randn(8)

    value_1 = critic(state_1)
    value_2 = critic(state_2)

    assert value_1.shape == (1,)
    assert value_2.shape == (1,)


def test_critic_gradients():
    critic = PPOCritic()

    states = torch.randn(8, 8)

    values = critic(states)

    loss = values.mean()

    loss.backward()

    for parameter in critic.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))