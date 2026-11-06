import numpy as np
import pytest

from src.environment.cloud_env import CloudAutoScalingEnv


def test_environment_reset():
    env = CloudAutoScalingEnv()

    state = env.reset()

    assert isinstance(state, np.ndarray)
    assert state.shape == (8,)


def test_action_space():
    env = CloudAutoScalingEnv()

    assert env.get_action_size() == 3
    assert env.get_state_size() == 8


def test_maintain_action():
    env = CloudAutoScalingEnv()

    env.reset()

    initial_replicas = env.simulator.replicas

    state, reward, done, info = env.step(0)

    assert env.simulator.replicas == initial_replicas
    assert state.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_scale_up_action():
    env = CloudAutoScalingEnv()

    env.reset()

    initial_replicas = env.simulator.replicas

    env.step(1)

    assert env.simulator.replicas == initial_replicas + 1


def test_scale_down_action():
    env = CloudAutoScalingEnv()

    env.reset()

    initial_replicas = env.simulator.replicas

    env.step(2)

    assert env.simulator.replicas == initial_replicas - 1


def test_invalid_action():
    env = CloudAutoScalingEnv()

    env.reset()

    with pytest.raises(ValueError):
        env.step(10)