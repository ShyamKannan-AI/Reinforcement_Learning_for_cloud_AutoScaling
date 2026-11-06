import torch
import torch.nn as nn

class DQNNetwork(nn.Module):
    """
    Neural network used by the Deep Q-Network agent.

    Input:
        8-dimensional state vector.

    Output:
        3 Q-values corresponding to:
            0 -> Maintain
            1 -> Scale up
            2 -> Scale down
    """
    def __init__(self, state_size=8, hidden_size=128, action_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    def forward(self, state):
        return self.network(state)
    