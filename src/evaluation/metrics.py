import numpy as np


def average_reward(records):
    rewards = [
        record["reward"]
        for record in records
    ]

    return float(np.mean(rewards))


def average_cpu_utilization(records):
    values = [
        record["cpu_utilization"]
        for record in records
    ]

    return float(np.mean(values))


def average_memory_utilization(records):
    values = [
        record["memory_utilization"]
        for record in records
    ]

    return float(np.mean(values))


def average_latency(records):
    values = [
        record["latency"]
        for record in records
    ]

    return float(np.mean(values))


def p95_latency(records):
    values = [
        record["latency"]
        for record in records
    ]

    return float(np.percentile(values, 95))


def total_cost(records):
    values = [
        record["cost"]
        for record in records
    ]

    return float(np.sum(values))


def sla_violation_count(records):
    return int(
        sum(
            record["sla_violation"]
            for record in records
        )
    )


def sla_violation_rate(records):
    if not records:
        return 0.0

    return float(
        sla_violation_count(records)
        / len(records)
    )


def average_replicas(records):
    values = [
        record["replicas"]
        for record in records
    ]

    return float(np.mean(values))


def peak_replicas(records):
    values = [
        record["replicas"]
        for record in records
    ]

    return int(np.max(values))


def scaling_events(records):
    """
    Count changes in replica count.
    """

    if len(records) < 2:
        return 0

    events = 0

    previous_replicas = records[0]["replicas"]

    for record in records[1:]:
        current_replicas = record["replicas"]

        if current_replicas != previous_replicas:
            events += 1

        previous_replicas = current_replicas

    return events


def calculate_metrics(records):
    """
    Calculate all standardized metrics for one episode.
    """

    return {
        "average_reward": average_reward(records),
        "average_cpu_utilization": average_cpu_utilization(
            records
        ),
        "average_memory_utilization": average_memory_utilization(
            records
        ),
        "average_latency": average_latency(records),
        "p95_latency": p95_latency(records),
        "total_cost": total_cost(records),
        "sla_violation_count": sla_violation_count(
            records
        ),
        "sla_violation_rate": sla_violation_rate(
            records
        ),
        "average_replicas": average_replicas(
            records
        ),
        "peak_replicas": peak_replicas(
            records
        ),
        "scaling_events": scaling_events(
            records
        ),
    }


def calculate_evaluation_metrics(all_records):
    """
    Calculate average metrics across multiple episodes.
    """

    episode_metrics = [
        calculate_metrics(records)
        for records in all_records
    ]

    metric_names = episode_metrics[0].keys()

    return {
        metric: float(
            np.mean(
                [
                    episode[metric]
                    for episode in episode_metrics
                ]
            )
        )
        for metric in metric_names
    }