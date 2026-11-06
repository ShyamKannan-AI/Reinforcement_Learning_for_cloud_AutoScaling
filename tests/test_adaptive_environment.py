from src.environment.adaptive_environment import (
    AdaptiveScalingEnvironment,
)
from src.environment.cloud_simulator import CloudSimulator
from src.controller.scaling_controller import ScalingController


def create_environment():
    simulator = CloudSimulator(
        max_steps=10,
        seed=42,
    )

    controller = ScalingController()

    return AdaptiveScalingEnvironment(
        simulator=simulator,
        controller=controller,
    )


def test_environment_reset():

    env = create_environment()

    state = env.reset()

    assert state is not None
    assert len(state) == 8


def test_environment_returns_current_info():

    env = create_environment()

    env.reset()

    info = env.get_current_info()

    assert "request_rate" in info
    assert "previous_request_rate" in info
    assert "cpu_utilization" in info
    assert "memory_utilization" in info
    assert "latency" in info
    assert "replicas" in info


def test_dqn_action_changes_replicas():

    env = create_environment()

    env.reset()

    initial_replicas = env.simulator.replicas

    env.apply_dqn_action(1)

    assert (
        env.simulator.replicas
        == initial_replicas + 1
    )


def test_ppo_cpu_action_changes_cpu():

    env = create_environment()

    env.reset()

    initial_cpu = env.simulator.cpu_per_replica

    env.apply_ppo_action(1)

    assert (
        env.simulator.cpu_per_replica
        == initial_cpu + 1.0
    )


def test_ppo_memory_action_changes_memory():

    env = create_environment()

    env.reset()

    initial_memory = (
        env.simulator.memory_per_replica
    )

    env.apply_ppo_action(3)

    assert (
        env.simulator.memory_per_replica
        == initial_memory + 2.0
    )


def test_environment_step_advances_once():

    env = create_environment()

    env.reset()

    initial_step = env.simulator.step_count

    next_state, reward, done, info = env.step()

    assert env.simulator.step_count == initial_step + 1

    assert next_state is not None
    assert isinstance(reward, float)
    assert isinstance(done, bool)

    assert "scaling_mode" in info
    assert "dqn_action" in info
    assert "ppo_action" in info