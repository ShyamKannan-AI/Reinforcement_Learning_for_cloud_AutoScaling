import numpy as np

from src.agents.ppo.rollout_buffer import PPORolloutBuffer


def create_state(value):
    return np.full(
        8,
        value,
        dtype=np.float32,
    )


def test_buffer_starts_empty():
    buffer = PPORolloutBuffer()

    assert len(buffer) == 0


def test_add_transition():
    buffer = PPORolloutBuffer()

    state = create_state(0.1)
    next_state = create_state(0.2)

    buffer.add(
        state=state,
        action=1,
        reward=2.5,
        next_state=next_state,
        done=False,
        log_probability=-0.75,
        value=3.2,
    )

    assert len(buffer) == 1


def test_multiple_transitions():
    buffer = PPORolloutBuffer()

    for i in range(5):
        buffer.add(
            state=create_state(i),
            action=i % 5,
            reward=float(i),
            next_state=create_state(i + 1),
            done=False,
            log_probability=-0.1 * i,
            value=float(i + 1),
        )

    assert len(buffer) == 5


def test_get_data_shapes():
    buffer = PPORolloutBuffer()

    for i in range(10):
        buffer.add(
            state=create_state(i),
            action=i % 5,
            reward=float(i),
            next_state=create_state(i + 1),
            done=False,
            log_probability=-0.1 * i,
            value=float(i + 1),
        )

    data = buffer.get_data()

    assert data["states"].shape == (10, 8)
    assert data["actions"].shape == (10,)
    assert data["rewards"].shape == (10,)
    assert data["next_states"].shape == (10, 8)
    assert data["dones"].shape == (10,)
    assert data["log_probabilities"].shape == (10,)
    assert data["values"].shape == (10,)


def test_data_types():
    buffer = PPORolloutBuffer()

    buffer.add(
        state=create_state(0.1),
        action=2,
        reward=1.0,
        next_state=create_state(0.2),
        done=False,
        log_probability=-0.5,
        value=2.0,
    )

    data = buffer.get_data()

    assert data["states"].dtype == np.float32
    assert data["actions"].dtype == np.int64
    assert data["rewards"].dtype == np.float32
    assert data["next_states"].dtype == np.float32
    assert data["dones"].dtype == np.float32
    assert data["log_probabilities"].dtype == np.float32
    assert data["values"].dtype == np.float32


def test_clear():
    buffer = PPORolloutBuffer()

    buffer.add(
        state=create_state(0.1),
        action=1,
        reward=1.0,
        next_state=create_state(0.2),
        done=False,
        log_probability=-0.5,
        value=2.0,
    )

    assert len(buffer) == 1

    buffer.clear()

    assert len(buffer) == 0

    data = buffer.get_data()

    assert data["states"].shape == (0, 8)
    assert data["actions"].shape == (0,)
    assert data["rewards"].shape == (0,)