import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.ppo.actor import PPOActor
from src.agents.ppo.critic import PPOCritic
from src.agents.ppo.rollout_buffer import PPORolloutBuffer
from src.agents.ppo.returns import (
    calculate_returns,
    calculate_gae,
    normalize_advantages,
)


class PPOAgent:
    """
    PPO agent for vertical scaling.

    The agent controls:
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
        actor_learning_rate=3e-4,
        critic_learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        update_epochs=10,
        batch_size=64,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

        self.entropy_coefficient = (
            entropy_coefficient
        )

        self.value_coefficient = (
            value_coefficient
        )

        self.update_epochs = update_epochs
        self.batch_size = batch_size

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.actor = PPOActor(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
        ).to(self.device)

        self.critic = PPOCritic(
            state_size=state_size,
            hidden_size=hidden_size,
        ).to(self.device)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=actor_learning_rate,
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=critic_learning_rate,
        )

        self.rollout_buffer = PPORolloutBuffer(
            state_size=state_size
        )

        self.loss_function = nn.MSELoss()

    def select_action(
        self,
        state,
        training=True,
    ):
        """
        Select an action using the Actor.

        During training:
            Sample from the policy distribution.

        During evaluation:
            Select the highest-probability action.

        Returns:
            action
            log_probability
            value
        """

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            if training:
                action, log_probability = (
                    self.actor.sample_action(
                        state_tensor
                    )
                )
            else:
                action = (
                    self.actor.deterministic_action(
                        state_tensor
                    )
                )

                distribution = (
                    self.actor.get_distribution(
                        state_tensor
                    )
                )

                log_probability = (
                    distribution.log_prob(
                        action
                    )
                )

            value = self.critic(
                state_tensor
            )

        return (
            int(action.item()),
            float(log_probability.item()),
            float(value.item()),
        )

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        log_probability,
        value,
    ):
        """
        Store one transition in the rollout buffer.
        """

        self.rollout_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_probability=log_probability,
            value=value,
        )

    def update(
        self,
        next_value=0.0,
    ):
        """
        Perform PPO optimization using the collected rollout.
        """

        if len(self.rollout_buffer) == 0:
            return None

        data = self.rollout_buffer.get_data()

        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        old_log_probabilities = (
            data["log_probabilities"]
        )
        values = data["values"]

        returns = calculate_returns(
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
            next_value=next_value,
        )

        advantages = calculate_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            next_value=next_value,
        )

        advantages = normalize_advantages(
            advantages
        )

        states_tensor = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions_tensor = torch.tensor(
            actions,
            dtype=torch.int64,
            device=self.device,
        )

        old_log_probabilities_tensor = (
            torch.tensor(
                old_log_probabilities,
                dtype=torch.float32,
                device=self.device,
            )
        )

        returns_tensor = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        advantages_tensor = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=self.device,
        )

        dataset_size = len(states_tensor)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        for _ in range(self.update_epochs):

            indices = np.random.permutation(
                dataset_size
            )

            for start in range(
                0,
                dataset_size,
                self.batch_size,
            ):
                end = min(
                    start + self.batch_size,
                    dataset_size,
                )

                batch_indices = indices[
                    start:end
                ]

                batch_indices = torch.tensor(
                    batch_indices,
                    dtype=torch.long,
                    device=self.device,
                )

                batch_states = (
                    states_tensor[batch_indices]
                )

                batch_actions = (
                    actions_tensor[batch_indices]
                )

                batch_old_log_probs = (
                    old_log_probabilities_tensor[
                        batch_indices
                    ]
                )

                batch_returns = (
                    returns_tensor[
                        batch_indices
                    ]
                )

                batch_advantages = (
                    advantages_tensor[
                        batch_indices
                    ]
                )

                new_log_probs, entropy = (
                    self.actor.evaluate_actions(
                        batch_states,
                        batch_actions,
                    )
                )

                ratio = torch.exp(
                    new_log_probs
                    - batch_old_log_probs
                )

                unclipped_objective = (
                    ratio * batch_advantages
                )

                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                )

                clipped_objective = (
                    clipped_ratio
                    * batch_advantages
                )

                actor_loss = -torch.min(
                    unclipped_objective,
                    clipped_objective,
                ).mean()

                entropy_bonus = entropy.mean()

                actor_loss = (
                    actor_loss
                    - self.entropy_coefficient
                    * entropy_bonus
                )

                predicted_values = (
                    self.critic(batch_states)
                    .squeeze(-1)
                )

                critic_loss = (
                    self.loss_function(
                        predicted_values,
                        batch_returns,
                    )
                )

                self.actor_optimizer.zero_grad()

                actor_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=0.5,
                )

                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()

                critic_total_loss = (
                    self.value_coefficient
                    * critic_loss
                )

                critic_total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    max_norm=0.5,
                )

                self.critic_optimizer.step()

                total_actor_loss += (
                    float(actor_loss.item())
                )

                total_critic_loss += (
                    float(critic_loss.item())
                )

                total_entropy += (
                    float(entropy_bonus.item())
                )

                update_count += 1

        self.rollout_buffer.clear()

        if update_count == 0:
            return None

        return {
            "actor_loss": (
                total_actor_loss
                / update_count
            ),
            "critic_loss": (
                total_critic_loss
                / update_count
            ),
            "entropy": (
                total_entropy
                / update_count
            ),
        }

    def save(self, path):
        """
        Save Actor, Critic and optimizer state.
        """

        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": (
                    self.actor_optimizer.state_dict()
                ),
                "critic_optimizer": (
                    self.critic_optimizer.state_dict()
                ),
            },
            path,
        )

    def load(self, path):
        """
        Load Actor, Critic and optimizer state.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.actor.load_state_dict(
            checkpoint["actor"]
        )

        self.critic.load_state_dict(
            checkpoint["critic"]
        )

        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer"]
        )

        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )