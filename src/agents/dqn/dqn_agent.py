import torch
import torch.nn as nn
import torch.optim as optim

import random

from src.agents.dqn.network import DQNNetwork
from src.agents.dqn.replay_buffer import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Network agent for horizontal cloud scaling.

    Actions:
        0 -> Maintain replicas
        1 -> Scale replicas up
        2 -> Scale replicas down
    """

    def __init__(
        self,
        state_size=8,
        action_size=3,
        hidden_size=128,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_frequency=100,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Online Q-network
        self.q_network = DQNNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
        ).to(self.device)

        # Target Q-network
        self.target_network = DQNNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
        ).to(self.device)

        # Initially both networks are identical.
        self.update_target_network()

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
        )

        self.loss_function = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity
        )

        self.training_steps = 0

    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy exploration.
        """

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        action = torch.argmax(q_values, dim=1).item()

        return action

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Store an experience in the replay buffer.
        """

        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def train_step(self):
        """
        Perform one DQN training step using a random
        minibatch from the replay buffer.
        """

        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.int64,
            device=self.device,
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        # Current Q-values for the selected actions.
        current_q_values = self.q_network(states)

        current_q_values = current_q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        # Calculate target Q-values.
        with torch.no_grad():
            next_q_values = self.target_network(next_states)

            max_next_q_values = next_q_values.max(
                dim=1
            ).values

            target_q_values = (
                rewards
                + self.gamma
                * max_next_q_values
                * (1.0 - dones)
            )

        # Calculate loss.
        loss = self.loss_function(
            current_q_values,
            target_q_values,
        )

        # Backpropagation.
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.training_steps += 1

        # Periodically update target network.
        if (
            self.training_steps
            % self.target_update_frequency
            == 0
        ):
            self.update_target_network()

        # Decay epsilon after training.
        self.decay_epsilon()

        return float(loss.item())

    def update_target_network(self):
        """
        Copy the online network weights into the target network.
        """

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

    def decay_epsilon(self):
        """
        Gradually reduce exploration.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path):
        """
        Save the trained DQN agent.
        """

        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps,
            },
            path,
        )

    def load(self, path):
        """
        Load a previously saved DQN agent.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.q_network.load_state_dict(
            checkpoint["q_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint["epsilon"]

        self.training_steps = checkpoint["training_steps"]