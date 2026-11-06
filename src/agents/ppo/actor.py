import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOActor(nn.Module):
    """
    Actor network for PPO vertical scaling.

    Input:
        8-dimensional cloud state.

    Output:
        Probability distribution over 5 vertical-scaling actions.

    Actions:
        0 -> Maintain
        1 -> Increase CPU
        2 -> Decrease CPU
        3 -> Increase memory
        4 -> Decrease memory
    """

    def __init__(
        self,
        state_size=8,
        action_size=5,
        hidden_size=128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                state_size,
                hidden_size,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_size,
                action_size,
            ),
        )

    def forward(self, state):
        """
        Return action logits.
        """

        return self.network(state)

    def get_distribution(self, state):
        """
        Return a categorical probability distribution
        over actions.
        """

        logits = self.forward(state)

        return Categorical(
            logits=logits
        )

    def sample_action(self, state):
        """
        Sample an action from the policy.

        Returns:
            action
            log probability
        """

        distribution = self.get_distribution(
            state
        )

        action = distribution.sample()

        log_probability = distribution.log_prob(
            action
        )

        return action, log_probability

    def evaluate_actions(
        self,
        state,
        actions,
    ):
        """
        Evaluate selected actions under the current policy.

        Returns:
            log probabilities
            entropy
        """

        distribution = self.get_distribution(
            state
        )

        log_probabilities = distribution.log_prob(
            actions
        )

        entropy = distribution.entropy()

        return (
            log_probabilities,
            entropy,
        )

    def deterministic_action(self, state):
        """
        Select the highest-probability action.

        Used during evaluation rather than exploration.
        """

        distribution = self.get_distribution(
            state
        )

        return torch.argmax(
            distribution.probs,
            dim=-1,
        )