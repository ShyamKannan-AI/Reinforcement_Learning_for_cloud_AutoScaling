import numpy as np
import torch

from src.agents.ppo.ppo_agent import PPOAgent


def create_state(value=0.5):
    return np.full(
        8,
        value,
        dtype=np.float32,
    )


def test_agent_initialization():
    agent = PPOAgent()

    assert agent.state_size == 8
    assert agent.action_size == 5
    assert agent.actor is not None
    assert agent.critic is not None
    assert agent.rollout_buffer is not None


def test_select_action_training():
    agent = PPOAgent()

    state = create_state()

    action, log_probability, value = (
        agent.select_action(
            state,
            training=True,
        )
    )

    assert 0 <= action < 5
    assert np.isfinite(log_probability)
    assert np.isfinite(value)


def test_select_action_evaluation():
    agent = PPOAgent()

    state = create_state()

    action, log_probability, value = (
        agent.select_action(
            state,
            training=False,
        )
    )

    assert 0 <= action < 5
    assert np.isfinite(log_probability)
    assert np.isfinite(value)


def test_remember():
    agent = PPOAgent()

    state = create_state(0.1)
    next_state = create_state(0.2)

    agent.remember(
        state=state,
        action=1,
        reward=2.0,
        next_state=next_state,
        done=False,
        log_probability=-0.5,
        value=1.5,
    )

    assert len(agent.rollout_buffer) == 1


def test_update_empty_buffer():
    agent = PPOAgent()

    result = agent.update()

    assert result is None


def test_update():
    agent = PPOAgent(
        update_epochs=2,
        batch_size=4,
    )

    for step in range(8):

        state = create_state(
            0.1 + step * 0.01
        )

        next_state = create_state(
            0.2 + step * 0.01
        )

        action, log_probability, value = (
            agent.select_action(
                state,
                training=True,
            )
        )

        agent.remember(
            state=state,
            action=action,
            reward=1.0,
            next_state=next_state,
            done=False,
            log_probability=log_probability,
            value=value,
        )

    result = agent.update(
        next_value=0.5
    )

    assert result is not None

    assert "actor_loss" in result
    assert "critic_loss" in result
    assert "entropy" in result

    assert np.isfinite(
        result["actor_loss"]
    )

    assert np.isfinite(
        result["critic_loss"]
    )

    assert np.isfinite(
        result["entropy"]
    )

    assert len(agent.rollout_buffer) == 0


def test_network_parameters_change_after_update():
    agent = PPOAgent(
        update_epochs=2,
        batch_size=4,
    )

    actor_before = [
        parameter.detach().clone()
        for parameter in agent.actor.parameters()
    ]

    critic_before = [
        parameter.detach().clone()
        for parameter in agent.critic.parameters()
    ]

    for step in range(8):

        state = create_state(
            0.1 + step * 0.01
        )

        next_state = create_state(
            0.2 + step * 0.01
        )

        action, log_probability, value = (
            agent.select_action(
                state,
                training=True,
            )
        )

        agent.remember(
            state=state,
            action=action,
            reward=1.0,
            next_state=next_state,
            done=False,
            log_probability=log_probability,
            value=value,
        )

    agent.update(
        next_value=0.5
    )

    actor_changed = any(
        not torch.equal(
            before,
            after,
        )
        for before, after in zip(
            actor_before,
            agent.actor.parameters(),
        )
    )

    critic_changed = any(
        not torch.equal(
            before,
            after,
        )
        for before, after in zip(
            critic_before,
            agent.critic.parameters(),
        )
    )

    assert actor_changed
    assert critic_changed