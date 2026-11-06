import numpy as np
import torch

from src.agents.ppo.actor import PPOActor


def create_state():
    return torch.tensor(
        np.array(
            [
                0.5,
                0.6,
                0.4,
                0.2,
                0.15,
                0.25,
                0.125,
                0.05,
            ],
            dtype=np.float32,
        )
    )


def test_actor_output_shape():

    actor = PPOActor()

    state = create_state()

    logits = actor(state)

    assert logits.shape == (5,)


def test_actor_batch_output_shape():

    actor = PPOActor()

    states = torch.randn(
        16,
        8,
    )

    logits = actor(states)

    assert logits.shape == (
        16,
        5,
    )


def test_distribution():

    actor = PPOActor()

    state = create_state()

    distribution = actor.get_distribution(
        state
    )

    probabilities = distribution.probs

    assert probabilities.shape == (5,)

    assert torch.all(
        probabilities >= 0
    )

    assert torch.isclose(
        probabilities.sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )


def test_sample_action():

    actor = PPOActor()

    state = create_state()

    action, log_probability = (
        actor.sample_action(state)
    )

    assert action.shape == torch.Size([])

    assert 0 <= action.item() < 5

    assert torch.isfinite(
        log_probability
    )


def test_evaluate_actions():

    actor = PPOActor()

    states = torch.randn(
        8,
        8,
    )

    actions = torch.randint(
        0,
        5,
        (8,),
    )

    log_probabilities, entropy = (
        actor.evaluate_actions(
            states,
            actions,
        )
    )

    assert log_probabilities.shape == (8,)

    assert entropy.shape == (8,)

    assert torch.all(
        torch.isfinite(
            log_probabilities
        )
    )

    assert torch.all(
        torch.isfinite(entropy)
    )


def test_deterministic_action():

    actor = PPOActor()

    states = torch.randn(
        10,
        8,
    )

    actions = actor.deterministic_action(
        states
    )

    assert actions.shape == (10,)

    assert torch.all(
        (actions >= 0)
        & (actions < 5)
    )