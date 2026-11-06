import numpy as np
import pytest

from src.agents.dqn.replay_buffer import ReplayBuffer


def create_experience(value):
    state = np.full(8, value, dtype=np.float32)
    action = 1
    reward = float(value)
    next_state = np.full(8, value + 1, dtype=np.float32)
    done = False

    return state, action, reward, next_state, done


def test_initial_buffer_is_empty():
    buffer = ReplayBuffer(capacity=10)

    assert len(buffer) == 0


def test_add_experience():
    buffer = ReplayBuffer(capacity=10)

    experience = create_experience(1)

    buffer.add(*experience)

    assert len(buffer) == 1


def test_buffer_capacity():
    buffer = ReplayBuffer(capacity=3)

    for i in range(5):
        buffer.add(*create_experience(i))

    assert len(buffer) == 3


def test_sample_shape():
    buffer = ReplayBuffer(capacity=10)

    for i in range(5):
        buffer.add(*create_experience(i))

    states, actions, rewards, next_states, dones = buffer.sample(3)

    assert states.shape == (3, 8)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert next_states.shape == (3, 8)
    assert dones.shape == (3,)


def test_sample_contains_valid_data():
    buffer = ReplayBuffer(capacity=10)

    for i in range(5):
        buffer.add(*create_experience(i))

    states, actions, rewards, next_states, dones = buffer.sample(3)

    assert states.dtype == np.float32
    assert actions.dtype == np.int64
    assert rewards.dtype == np.float32
    assert next_states.dtype == np.float32
    assert dones.dtype == np.float32


def test_is_ready():
    buffer = ReplayBuffer(capacity=10)

    assert not buffer.is_ready(3)

    for i in range(3):
        buffer.add(*create_experience(i))

    assert buffer.is_ready(3)


def test_sampling_too_many_experiences():
    buffer = ReplayBuffer(capacity=10)

    buffer.add(*create_experience(1))

    with pytest.raises(ValueError):
        buffer.sample(2)


def test_clear():
    buffer = ReplayBuffer(capacity=10)

    for i in range(5):
        buffer.add(*create_experience(i))

    assert len(buffer) == 5

    buffer.clear()

    assert len(buffer) == 0