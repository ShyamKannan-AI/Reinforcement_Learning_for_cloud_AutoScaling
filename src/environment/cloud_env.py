from src.environment.cloud_simulator import CloudSimulator


class CloudAutoScalingEnv:
    """
    Custom environment for the DQN agent.

    The DQN agent is responsible only for horizontal scaling.

    Actions:
        0 -> Maintain current number of replicas
        1 -> Increase replicas by one
        2 -> Decrease replicas by one
    """

    MAINTAIN = 0
    SCALE_UP = 1
    SCALE_DOWN = 2

    def __init__(self, simulator=None):
        """
        Initialize the DQN environment.

        Args:
            simulator:
                Optional CloudSimulator instance.
                If not provided, a new simulator is created.
        """

        self.simulator = simulator or CloudSimulator()

        self.state_size = 8
        self.action_size = 3

    def reset(self):
        """
        Reset the environment.

        Returns:
            np.ndarray:
                Initial state.
        """

        return self.simulator.reset()

    def step(self, action):
        """
        Apply a DQN action and advance the simulation.

        Args:
            action:
                0 -> maintain
                1 -> scale up
                2 -> scale down

        Returns:
            state:
                New state.
            reward:
                Reward after the action.
            done:
                Whether the episode has ended.
            info:
                Additional simulation information.
        """

        self._apply_action(action)

        state, reward, done, info = self.simulator.step()

        return state, reward, done, info

    def _apply_action(self, action):
        """
        Convert a DQN action into a simulator scaling operation.
        """

        if action == self.MAINTAIN:
            self.simulator.scale_horizontal(0)

        elif action == self.SCALE_UP:
            self.simulator.scale_horizontal(1)

        elif action == self.SCALE_DOWN:
            self.simulator.scale_horizontal(-1)

        else:
            raise ValueError(
                f"Invalid DQN action: {action}. "
                f"Expected 0, 1, or 2."
            )

    def get_state_size(self):
        """
        Return the size of the state vector.
        """

        return self.state_size

    def get_action_size(self):
        """
        Return the number of possible DQN actions.
        """

        return self.action_size

    def get_current_resources(self):
        """
        Return the current cloud resource configuration.
        """

        return {
            "replicas": self.simulator.replicas,
            "cpu_per_replica": self.simulator.cpu_per_replica,
            "memory_per_replica": self.simulator.memory_per_replica,
        }