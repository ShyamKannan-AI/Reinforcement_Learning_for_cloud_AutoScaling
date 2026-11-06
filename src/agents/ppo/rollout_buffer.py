import numpy as np


class PPORolloutBuffer:
    """
    Stores rollout data collected by the PPO agent.

    Each transition contains:

        state
        action
        reward
        next_state
        done
        log_probability
        value

    The stored rollout is later used to calculate
    returns and advantages for PPO training.
    """

    def __init__(self, state_size=8):
        self.state_size = state_size

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probabilities = []
        self.values = []

    def add(
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
        Add one transition to the rollout buffer.
        """

        self.states.append(
            np.asarray(
                state,
                dtype=np.float32,
            )
        )

        self.actions.append(
            int(action)
        )

        self.rewards.append(
            float(reward)
        )

        self.next_states.append(
            np.asarray(
                next_state,
                dtype=np.float32,
            )
        )

        self.dones.append(
            bool(done)
        )

        self.log_probabilities.append(
            float(log_probability)
        )

        self.values.append(
            float(value)
        )

    def __len__(self):
        return len(self.states)

    def clear(self):
        """
        Remove all stored rollout data.
        """

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probabilities.clear()
        self.values.clear()

    def get_data(self):
        """
        Return the complete rollout as NumPy arrays.

        Empty state arrays retain the shape:
            (0, state_size)
        """

        if self.states:
            states = np.asarray(
                self.states,
                dtype=np.float32,
            )

            next_states = np.asarray(
                self.next_states,
                dtype=np.float32,
            )
        else:
            states = np.empty(
                (0, self.state_size),
                dtype=np.float32,
            )

            next_states = np.empty(
                (0, self.state_size),
                dtype=np.float32,
            )

        return {
            "states": states,
            "actions": np.asarray(
                self.actions,
                dtype=np.int64,
            ),
            "rewards": np.asarray(
                self.rewards,
                dtype=np.float32,
            ),
            "next_states": next_states,
            "dones": np.asarray(
                self.dones,
                dtype=np.float32,
            ),
            "log_probabilities": np.asarray(
                self.log_probabilities,
                dtype=np.float32,
            ),
            "values": np.asarray(
                self.values,
                dtype=np.float32,
            ),
        }