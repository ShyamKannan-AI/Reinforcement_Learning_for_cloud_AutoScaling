import torch
import torch.nn as nn


class PPOCritic(nn.Module):
    """
    Critic network for PPO vertical scaling.

    Input:
        8-dimensional cloud state.

    Output:
        A single scalar value V(s), representing the
        estimated expected future return from the state.
    """

    def __init__(
        self,
        state_size=8,
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
                1,
            ),
        )

    def forward(self, state):
        return self.network(state)