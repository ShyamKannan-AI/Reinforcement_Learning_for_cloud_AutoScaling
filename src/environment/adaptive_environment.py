from src.environment.cloud_simulator import CloudSimulator
from src.controller.scaling_controller import ScalingController


class AdaptiveScalingEnvironment:
    """
    Shared execution environment for the adaptive auto-scaling system.

    The environment coordinates:

        DQN
            Horizontal scaling
            Replica count

        PPO
            Vertical scaling
            CPU and memory allocation

        ScalingController
            Decides whether DQN, PPO, or both should be activated.

    Important:
        DQN and PPO operate on the same CloudSimulator.

        The simulator advances exactly once per environment step.
    """

    def __init__(
        self,
        simulator=None,
        controller=None,
    ):
        self.simulator = simulator or CloudSimulator()
        self.controller = controller or ScalingController()

        self.state_size = 8

        # DQN horizontal actions:
        # 0 = maintain
        # 1 = scale up
        # 2 = scale down
        self.dqn_action_size = 3

        # PPO vertical actions:
        # 0 = maintain
        # 1 = increase CPU
        # 2 = decrease CPU
        # 3 = increase memory
        # 4 = decrease memory
        self.ppo_action_size = 5

    def reset(self):
        """
        Reset the simulator and controller.

        Returns:
            Initial environment state.
        """

        self.controller.reset()

        return self.simulator.reset()

    def get_current_info(self):
        """
        Return the current simulator metrics before
        advancing the environment.

        This information is used by the adaptive controller
        to determine which scaling mode should be activated.
        """

        return {
            "request_rate": self.simulator.request_rate,
            "previous_request_rate": (
                self.simulator.previous_request_rate
            ),
            "cpu_utilization": (
                self.simulator.cpu_utilization
            ),
            "memory_utilization": (
                self.simulator.memory_utilization
            ),
            "latency": self.simulator.latency,
            "cost": self.simulator.cost,
            "sla_violation": self.simulator.sla_violation,
            "replicas": self.simulator.replicas,
            "cpu_per_replica": (
                self.simulator.cpu_per_replica
            ),
            "memory_per_replica": (
                self.simulator.memory_per_replica
            ),
        }

    def apply_dqn_action(self, action):
        """
        Apply a DQN horizontal-scaling action.

        Actions:
            0 -> Maintain replicas
            1 -> Increase replicas
            2 -> Decrease replicas
        """

        if action == 0:
            self.simulator.scale_horizontal(0)

        elif action == 1:
            self.simulator.scale_horizontal(1)

        elif action == 2:
            self.simulator.scale_horizontal(-1)

        else:
            raise ValueError(
                f"Invalid DQN action: {action}. "
                f"Expected 0, 1, or 2."
            )

    def apply_ppo_action(self, action):
        """
        Apply a PPO vertical-scaling action.

        Actions:
            0 -> Maintain resources
            1 -> Increase CPU
            2 -> Decrease CPU
            3 -> Increase memory
            4 -> Decrease memory
        """

        if action == 0:
            return

        elif action == 1:
            self.simulator.scale_cpu(1)

        elif action == 2:
            self.simulator.scale_cpu(-1)

        elif action == 3:
            self.simulator.scale_memory(1)

        elif action == 4:
            self.simulator.scale_memory(-1)

        else:
            raise ValueError(
                f"Invalid PPO action: {action}. "
                f"Expected 0, 1, 2, 3, or 4."
            )

    def step(
        self,
        dqn_agent=None,
        ppo_agent=None,
    ):
        """
        Execute one adaptive scaling timestep.

        The controller determines which agent(s) should act.

        The selected actions are applied to the shared simulator,
        and the simulator advances exactly once.

        Returns:
            next_state
            reward
            done
            info
        """

        # --------------------------------------------------
        # 1. Get current state
        # --------------------------------------------------

        state = self.simulator.get_state()

        # --------------------------------------------------
        # 2. Get current system metrics
        # --------------------------------------------------

        current_info = self.get_current_info()

        # --------------------------------------------------
        # 3. Ask controller which scaling mechanism to use
        # --------------------------------------------------

        mode = self.controller.select_mode(
            state=state,
            info=current_info,
        )

        dqn_action = 0
        ppo_action = 0

        # --------------------------------------------------
        # 4. Select DQN action if horizontal scaling needed
        # --------------------------------------------------

        if mode in (
            ScalingController.MODE_HORIZONTAL,
            ScalingController.MODE_BOTH,
        ):
            if dqn_agent is None:
                raise ValueError(
                    "DQN agent is required for horizontal scaling."
                )

            dqn_action = dqn_agent.select_action(
                state,
                training=False,
            )

            self.apply_dqn_action(dqn_action)

        # --------------------------------------------------
        # 5. Select PPO action if vertical scaling needed
        # --------------------------------------------------

        if mode in (
            ScalingController.MODE_VERTICAL,
            ScalingController.MODE_BOTH,
        ):
            if ppo_agent is None:
                raise ValueError(
                    "PPO agent is required for vertical scaling."
                )

            ppo_action, _, _ = ppo_agent.select_action(
                state,
                training=False,
            )

            self.apply_ppo_action(ppo_action)

        # --------------------------------------------------
        # 6. Advance simulator exactly once
        # --------------------------------------------------

        next_state, reward, done, info = (
            self.simulator.step()
        )

        # --------------------------------------------------
        # 7. Add adaptive-controller information
        # --------------------------------------------------

        info["scaling_mode"] = mode
        info["dqn_action"] = dqn_action
        info["ppo_action"] = ppo_action

        return (
            next_state,
            reward,
            done,
            info,
        )

    def get_state_size(self):
        """Return the shared state size."""

        return self.state_size

    def get_dqn_action_size(self):
        """Return the DQN action-space size."""

        return self.dqn_action_size

    def get_ppo_action_size(self):
        """Return the PPO action-space size."""

        return self.ppo_action_size

    def get_current_resources(self):
        """
        Return the current resource configuration.
        """

        return {
            "replicas": self.simulator.replicas,
            "cpu_per_replica": (
                self.simulator.cpu_per_replica
            ),
            "memory_per_replica": (
                self.simulator.memory_per_replica
            ),
        }