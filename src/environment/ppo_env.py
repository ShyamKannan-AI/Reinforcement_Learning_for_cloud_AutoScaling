from src.environment.cloud_simulator import CloudSimulator


class PPOVerticalScalingEnv:
    """
    Custom environment for the PPO vertical-scaling agent.

    The PPO agent controls CPU and memory allocation.

    Actions:
        0 -> Maintain resources
        1 -> Increase CPU
        2 -> Decrease CPU
        3 -> Increase memory
        4 -> Decrease memory

    Horizontal scaling is deliberately excluded.
    """

    MAINTAIN = 0
    INCREASE_CPU = 1
    DECREASE_CPU = 2
    INCREASE_MEMORY = 3
    DECREASE_MEMORY = 4

    def __init__(self, simulator=None):
        self.simulator = simulator or CloudSimulator()

        self.state_size = 8
        self.action_size = 5

    def reset(self):
        """
        Reset the simulator and return the initial state.
        """

        return self.simulator.reset()

    def step(self, action):
        """
        Apply a vertical-scaling action and advance
        the simulator by one timestep.
        """

        self._apply_action(action)

        state, reward, done, info = (
            self.simulator.step()
        )

        return state, reward, done, info

    def _apply_action(self, action):
        """
        Translate PPO action into simulator resource changes.
        """

        if action == self.MAINTAIN:

            # No vertical scaling.
            return

        elif action == self.INCREASE_CPU:

            self.simulator.scale_cpu(1)

        elif action == self.DECREASE_CPU:

            self.simulator.scale_cpu(-1)

        elif action == self.INCREASE_MEMORY:

            self.simulator.scale_memory(1)

        elif action == self.DECREASE_MEMORY:

            self.simulator.scale_memory(-1)

        else:

            raise ValueError(
                f"Invalid PPO action: {action}. "
                f"Expected 0, 1, 2, 3, or 4."
            )

    def get_state_size(self):
        """
        Return dimensionality of the state.
        """

        return self.state_size

    def get_action_size(self):
        """
        Return number of PPO actions.
        """

        return self.action_size

    def get_current_resources(self):
        """
        Return current vertical resource allocation.
        """

        return {
            "cpu_per_replica": (
                self.simulator.cpu_per_replica
            ),
            "memory_per_replica": (
                self.simulator.memory_per_replica
            ),
            "replicas": self.simulator.replicas,
        }