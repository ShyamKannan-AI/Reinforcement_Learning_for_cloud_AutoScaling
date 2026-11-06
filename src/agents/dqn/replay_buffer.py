from collections import deque
import random
import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer for the DQN agent.

    Each experience contains:

        state
        action
        reward
        next_state
        done
    """

    def __init__(self, capacity=10000):
        self.capacity = capacity

        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Store one experience in the replay buffer.
        """

        experience = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )

        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences.
        """

        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} experiences "
                f"from buffer containing {len(self.buffer)} experiences."
            )

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

    def __len__(self):
        """
        Return the current number of stored experiences.
        """

        return len(self.buffer)

    def is_ready(self, batch_size):
        """
        Check whether enough experiences are available
        for training.
        """

        return len(self.buffer) >= batch_size

    def clear(self):
        """
        Remove all experiences from the buffer.
        """

        self.buffer.clear()