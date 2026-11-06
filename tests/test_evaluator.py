from src.agents.dqn.dqn_agent import DQNAgent
from src.environment.cloud_env import CloudAutoScalingEnv
from src.environment.cloud_simulator import CloudSimulator
from src.evaluation.evaluator import DQNEvaluator


def create_evaluator():

    simulator = CloudSimulator(
        max_steps=10,
        seed=42,
    )

    env = CloudAutoScalingEnv(
        simulator=simulator
    )

    agent = DQNAgent()

    return DQNEvaluator(
        env,
        agent,
    )


def test_run_episode():

    evaluator = create_evaluator()

    reward, records = evaluator.run_episode(
        max_steps=5
    )

    assert isinstance(reward, float)
    assert len(records) == 5


def test_episode_records():

    evaluator = create_evaluator()

    _, records = evaluator.run_episode(
        max_steps=5
    )

    record = records[0]

    required_fields = [
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
    ]

    for field in required_fields:
        assert field in record


def test_multiple_episode_evaluation():

    evaluator = create_evaluator()

    results = evaluator.evaluate(
        num_episodes=3,
        max_steps=5,
    )

    assert len(
        results["episode_rewards"]
    ) == 3

    assert len(
        results["records"]
    ) == 3