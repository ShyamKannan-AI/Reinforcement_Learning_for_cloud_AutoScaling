import torch

from src.agents.dqn.network import DQNNetwork


def test_dqn_network_output_shape():
    network = DQNNetwork(
        state_size=8,
        action_size=3,
    )

    state = torch.randn(1, 8)

    q_values = network(state)

    assert q_values.shape == (1, 3)


def test_dqn_network_batch():
    network = DQNNetwork(
        state_size=8,
        action_size=3,
    )

    states = torch.randn(32, 8)

    q_values = network(states)

    assert q_values.shape == (32, 3)


def test_dqn_network_output_is_finite():
    network = DQNNetwork(
        state_size=8,
        action_size=3,
    )

    state = torch.randn(1, 8)

    q_values = network(state)

    assert torch.isfinite(q_values).all()


def test_dqn_network_has_trainable_parameters():
    network = DQNNetwork(
        state_size=8,
        action_size=3,
    )

    parameters = list(network.parameters())

    assert len(parameters) > 0

    assert all(
        parameter.requires_grad
        for parameter in parameters
    )