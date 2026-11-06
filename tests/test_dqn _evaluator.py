import os

from src.environment.cloud_simulator import CloudSimulator
from src.environment.cloud_env import CloudAutoScalingEnv
from src.agents.dqn.dqn_agent import DQNAgent
from src.evaluation.evaluator import DQNEvaluator


MODEL_PATH = "models/dqn_horizontal_scaler.pt"


def test_dqn_evaluator_runs():

    assert os.path.exists(
        MODEL_PATH
    )

    simulator = CloudSimulator(
        max_steps=10,
        seed=42,
    )

    env = CloudAutoScalingEnv(
        simulator=simulator
    )

    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
    )

    agent.load(
        MODEL_PATH
    )

    agent.epsilon = 0.0

    evaluator = DQNEvaluator(
        env=env,
        agent=agent,
    )

    results = evaluator.evaluate(
        num_episodes=1,
        max_steps=10,
    )

    assert "episode_rewards" in results
    assert "records" in results

    assert len(
        results["episode_rewards"]
    ) == 1

    assert len(
        results["records"]
    ) == 1

    assert len(
        results["records"][0]
    ) == 10


def test_dqn_evaluation_record():

    simulator = CloudSimulator(
        max_steps=5,
        seed=42,
    )

    env = CloudAutoScalingEnv(
        simulator=simulator
    )

    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
    )

    agent.load(
        MODEL_PATH
    )

    agent.epsilon = 0.0

    evaluator = DQNEvaluator(
        env=env,
        agent=agent,
    )

    results = evaluator.evaluate(
        num_episodes=1,
        max_steps=5,
    )

    record = results[
        "records"
    ][0][0]

    expected_keys = {
        "reward",
        "action",
        "request_rate",
        "cpu_utilization",
        "memory_utilization",
        "latency",
        "cost",
        "sla_violation",
        "replicas",
        "cpu_per_replica",
        "memory_per_replica",
    }

    assert expected_keys.issubset(
        record.keys()
    )

    assert 0 <= record["action"] <= 2

    assert record["replicas"] >= 1