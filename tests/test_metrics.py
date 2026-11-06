from src.evaluation.metrics import (
    average_reward,
    average_cpu_utilization,
    average_memory_utilization,
    average_latency,
    p95_latency,
    total_cost,
    sla_violation_count,
    sla_violation_rate,
    average_replicas,
    peak_replicas,
    scaling_events,
    calculate_metrics,
    calculate_evaluation_metrics,
)


def sample_records():
    return [
        {
            "reward": 1.0,
            "cpu_utilization": 40.0,
            "memory_utilization": 50.0,
            "latency": 100.0,
            "cost": 5.0,
            "sla_violation": 0,
            "replicas": 2,
        },
        {
            "reward": 2.0,
            "cpu_utilization": 60.0,
            "memory_utilization": 70.0,
            "latency": 150.0,
            "cost": 6.0,
            "sla_violation": 0,
            "replicas": 3,
        },
        {
            "reward": -1.0,
            "cpu_utilization": 80.0,
            "memory_utilization": 90.0,
            "latency": 300.0,
            "cost": 7.0,
            "sla_violation": 1,
            "replicas": 2,
        },
    ]


def test_average_reward():
    assert average_reward(sample_records()) == 2.0 / 3.0


def test_average_cpu_utilization():
    assert average_cpu_utilization(
        sample_records()
    ) == 60.0


def test_average_memory_utilization():
    assert average_memory_utilization(
        sample_records()
    ) == 70.0


def test_average_latency():
    assert average_latency(
        sample_records()
    ) == 550.0 / 3.0


def test_p95_latency():
    result = p95_latency(sample_records())

    assert result > 150.0
    assert result <= 300.0


def test_total_cost():
    assert total_cost(
        sample_records()
    ) == 18.0


def test_sla_violation_count():
    assert sla_violation_count(
        sample_records()
    ) == 1


def test_sla_violation_rate():
    assert sla_violation_rate(
        sample_records()
    ) == 1.0 / 3.0


def test_average_replicas():
    assert average_replicas(
        sample_records()
    ) == 7.0 / 3.0


def test_peak_replicas():
    assert peak_replicas(
        sample_records()
    ) == 3


def test_scaling_events():
    assert scaling_events(
        sample_records()
    ) == 2


def test_calculate_metrics():
    metrics = calculate_metrics(
        sample_records()
    )

    assert "average_reward" in metrics
    assert "p95_latency" in metrics
    assert "total_cost" in metrics
    assert "sla_violation_rate" in metrics
    assert "scaling_events" in metrics


def test_calculate_evaluation_metrics():
    records = sample_records()

    metrics = calculate_evaluation_metrics(
        [records, records]
    )

    assert metrics["average_reward"] == 2.0 / 3.0
    assert metrics["peak_replicas"] == 3