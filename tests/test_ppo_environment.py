import pytest

from src.environment.cloud_simulator import CloudSimulator
from src.environment.ppo_env import PPOVerticalScalingEnv


def create_environment():
    simulator = CloudSimulator(
        max_steps=10,
        seed=42,
    )

    return PPOVerticalScalingEnv(
        simulator=simulator
    )


def test_reset():

    env = create_environment()

    state = env.reset()

    assert state.shape == (8,)


def test_state_and_action_sizes():

    env = create_environment()

    assert env.get_state_size() == 8
    assert env.get_action_size() == 5


def test_maintain_action():

    env = create_environment()

    env.reset()

    original_cpu = (
        env.simulator.cpu_per_replica
    )

    original_memory = (
        env.simulator.memory_per_replica
    )

    env._apply_action(
        env.MAINTAIN
    )

    assert (
        env.simulator.cpu_per_replica
        == original_cpu
    )

    assert (
        env.simulator.memory_per_replica
        == original_memory
    )


def test_increase_cpu():

    env = create_environment()

    env.reset()

    original_cpu = (
        env.simulator.cpu_per_replica
    )

    env._apply_action(
        env.INCREASE_CPU
    )

    assert (
        env.simulator.cpu_per_replica
        == original_cpu + 1.0
    )


def test_decrease_cpu():

    env = create_environment()

    env.reset()

    original_cpu = (
        env.simulator.cpu_per_replica
    )

    env._apply_action(
        env.DECREASE_CPU
    )

    assert (
        env.simulator.cpu_per_replica
        == original_cpu - 1.0
    )


def test_increase_memory():

    env = create_environment()

    env.reset()

    original_memory = (
        env.simulator.memory_per_replica
    )

    env._apply_action(
        env.INCREASE_MEMORY
    )

    assert (
        env.simulator.memory_per_replica
        == original_memory + 2.0
    )


def test_decrease_memory():

    env = create_environment()

    env.reset()

    original_memory = (
        env.simulator.memory_per_replica
    )

    env._apply_action(
        env.DECREASE_MEMORY
    )

    assert (
        env.simulator.memory_per_replica
        == original_memory - 2.0
    )


def test_horizontal_scaling_is_not_changed():

    env = create_environment()

    env.reset()

    original_replicas = (
        env.simulator.replicas
    )

    env._apply_action(
        env.INCREASE_CPU
    )

    assert (
        env.simulator.replicas
        == original_replicas
    )


def test_step():

    env = create_environment()

    env.reset()

    state, reward, done, info = env.step(
        env.INCREASE_CPU
    )

    assert state.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)

    assert "cpu_per_replica" in info
    assert "memory_per_replica" in info


def test_invalid_action():

    env = create_environment()

    env.reset()

    with pytest.raises(ValueError):

        env.step(10)