import numpy as np
import torch
import random

from src.agents.dqn.dqn_agent import DQNAgent


def create_state(value=0.5):
    return np.full(
        8,
        value,
        dtype=np.float32,
    )


def test_agent_initialization():
    agent = DQNAgent()

    assert agent.state_size == 8
    assert agent.action_size == 3
    assert agent.epsilon == 1.0
    assert len(agent.replay_buffer) == 0


def test_action_is_valid():
    agent = DQNAgent()

    state = create_state()

    action = agent.select_action(
        state,
        training=False,
    )

    assert action in [0, 1, 2]


def test_random_action_during_exploration():
    agent = DQNAgent(
        epsilon=1.0
    )

    state = create_state()

    action = agent.select_action(
        state,
        training=True,
    )

    assert action in [0, 1, 2]


def test_remember():
    agent = DQNAgent()

    state = create_state(0.1)
    next_state = create_state(0.2)

    agent.remember(
        state,
        1,
        2.0,
        next_state,
        False,
    )

    assert len(agent.replay_buffer) == 1


def test_training_waits_for_batch():
    agent = DQNAgent(
        batch_size=4
    )

    state = create_state()

    loss = agent.train_step()

    assert loss is None


def test_training_step():
    agent = DQNAgent(
        batch_size=4
    )

    for i in range(4):
        state = create_state(i / 10.0)
        next_state = create_state(
            (i + 1) / 10.0
        )

        agent.remember(
            state,
            i % 3,
            float(i),
            next_state,
            False,
        )

    loss = agent.train_step()

    assert loss is not None
    assert np.isfinite(loss)


def test_target_network_initially_matches_online_network():
    agent = DQNAgent()

    for online_param, target_param in zip(
        agent.q_network.parameters(),
        agent.target_network.parameters(),
    ):
        assert torch.equal(
            online_param,
            target_param,
        )


def test_epsilon_decay():
    agent = DQNAgent(
        epsilon=1.0,
        epsilon_decay=0.5,
        epsilon_min=0.1,
    )

    agent.decay_epsilon()

    assert agent.epsilon == 0.5

    agent.decay_epsilon()

    assert agent.epsilon == 0.25


def test_epsilon_does_not_go_below_minimum():
    agent = DQNAgent(
        epsilon=0.2,
        epsilon_decay=0.1,
        epsilon_min=0.1,
    )

    agent.decay_epsilon()

    assert agent.epsilon == 0.1