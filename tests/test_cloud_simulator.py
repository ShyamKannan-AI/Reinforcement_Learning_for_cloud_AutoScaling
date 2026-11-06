import numpy as np

from src.environment.cloud_simulator import CloudSimulator


def test_simulator_reset():
    simulator = CloudSimulator(seed=42)

    state = simulator.reset()

    assert isinstance(state, np.ndarray)
    assert state.shape == (8,)
    assert simulator.replicas == 2
    assert simulator.cpu_per_replica == 2.0
    assert simulator.memory_per_replica == 4.0


def test_horizontal_scaling():
    simulator = CloudSimulator(seed=42)

    simulator.reset()

    simulator.scale_horizontal(1)

    assert simulator.replicas == 3

    simulator.scale_horizontal(-1)

    assert simulator.replicas == 2


def test_horizontal_scaling_limits():
    simulator = CloudSimulator(seed=42)

    simulator.reset()

    for _ in range(100):
        simulator.scale_horizontal(1)

    assert simulator.replicas == simulator.max_replicas

    for _ in range(100):
        simulator.scale_horizontal(-1)

    assert simulator.replicas == simulator.min_replicas


def test_cpu_scaling():
    simulator = CloudSimulator(seed=42)

    simulator.reset()

    simulator.scale_cpu(1)

    assert simulator.cpu_per_replica == 3.0

    simulator.scale_cpu(-1)

    assert simulator.cpu_per_replica == 2.0


def test_memory_scaling():
    simulator = CloudSimulator(seed=42)

    simulator.reset()

    simulator.scale_memory(1)

    assert simulator.memory_per_replica == 6.0

    simulator.scale_memory(-1)

    assert simulator.memory_per_replica == 4.0


def test_simulation_step():
    simulator = CloudSimulator(seed=42)

    simulator.reset()

    state, reward, terminated, info = simulator.step()

    assert state.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)

    assert "request_rate" in info
    assert "cpu_utilization" in info
    assert "memory_utilization" in info
    assert "latency" in info
    assert "cost" in info


def test_state_is_finite():
    simulator = CloudSimulator(seed=42)

    state = simulator.reset()

    assert np.all(np.isfinite(state))