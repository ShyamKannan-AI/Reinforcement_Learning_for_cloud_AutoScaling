from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "dqn_evaluation.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "dqn_behavior"
)


def load_evaluation_data():
    """
    Load detailed DQN evaluation records.
    """

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {RESULTS_PATH}"
        )

    data = np.load(
        RESULTS_PATH,
        allow_pickle=True,
    )

    records = data["records"]

    return records


def plot_workload_and_replicas(records, output_dir):
    """
    Plot workload against the number of replicas.
    """

    episode = records[0]

    steps = list(range(1, len(episode) + 1))

    workload = [
        record["request_rate"]
        for record in episode
    ]

    replicas = [
        record["replicas"]
        for record in episode
    ]

    fig, ax1 = plt.subplots(
        figsize=(12, 6)
    )

    ax1.plot(
        steps,
        workload,
        label="Request Rate",
    )

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Request Rate")

    ax2 = ax1.twinx()

    ax2.plot(
        steps,
        replicas,
        label="Replicas",
    )

    ax2.set_ylabel("Replicas")

    ax1.set_title(
        "DQN Horizontal Scaling: Workload vs Replicas"
    )

    fig.tight_layout()

    output_path = (
        output_dir
        / "dqn_workload_vs_replicas.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_utilization(records, output_dir):
    """
    Plot CPU and memory utilization.
    """

    episode = records[0]

    steps = list(range(1, len(episode) + 1))

    cpu = [
        record["cpu_utilization"]
        for record in episode
    ]

    memory = [
        record["memory_utilization"]
        for record in episode
    ]

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        steps,
        cpu,
        label="CPU Utilization",
    )

    ax.plot(
        steps,
        memory,
        label="Memory Utilization",
    )

    ax.axhline(
        70.0,
        linestyle="--",
        label="70% Reference",
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Utilization (%)")

    ax.set_title(
        "DQN Resource Utilization"
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        output_dir
        / "dqn_utilization.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_latency(records, output_dir):
    """
    Plot application latency.
    """

    episode = records[0]

    steps = list(range(1, len(episode) + 1))

    latency = [
        record["latency"]
        for record in episode
    ]

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        steps,
        latency,
        label="Latency",
    )

    ax.axhline(
        200.0,
        linestyle="--",
        label="SLA Latency",
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Latency (ms)")

    ax.set_title(
        "DQN Application Latency"
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        output_dir
        / "dqn_latency.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_actions(records, output_dir):
    """
    Plot DQN horizontal scaling actions.

    Actions:
        0 = Maintain
        1 = Scale Up
        2 = Scale Down
    """

    episode = records[0]

    steps = list(range(1, len(episode) + 1))

    actions = [
        record["action"]
        for record in episode
    ]

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.step(
        steps,
        actions,
        where="post",
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Action")

    ax.set_yticks(
        [0, 1, 2],
        [
            "Maintain",
            "Scale Up",
            "Scale Down",
        ],
    )

    ax.set_title(
        "DQN Horizontal Scaling Actions"
    )

    fig.tight_layout()

    output_path = (
        output_dir
        / "dqn_actions.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():

    print("\nLoading DQN evaluation results...")

    records = load_evaluation_data()

    if len(records) == 0:
        raise ValueError(
            "No evaluation records found."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Evaluation episodes:",
        len(records),
    )

    print(
        "Steps in first episode:",
        len(records[0]),
    )

    plot_workload_and_replicas(
        records,
        OUTPUT_DIR,
    )

    plot_utilization(
        records,
        OUTPUT_DIR,
    )

    plot_latency(
        records,
        OUTPUT_DIR,
    )

    plot_actions(
        records,
        OUTPUT_DIR,
    )

    print("\nDQN behavior visualization completed.")
    print(
        "Plots saved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()