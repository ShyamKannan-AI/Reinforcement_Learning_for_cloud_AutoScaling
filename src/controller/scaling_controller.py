class ScalingController:
    """
    Deterministic adaptive controller that decides which
    scaling mechanism should be activated.

    Modes:

        "none"
            No scaling required.

        "horizontal"
            Activate DQN for replica scaling.

        "vertical"
            Activate PPO for CPU/memory scaling.

        "both"
            Activate both DQN and PPO.

    The controller does not directly modify the simulator.
    It only decides which RL agent(s) should be used.
    """

    MODE_NONE = "none"
    MODE_HORIZONTAL = "horizontal"
    MODE_VERTICAL = "vertical"
    MODE_BOTH = "both"

    def __init__(
        self,
        cpu_scale_threshold=70.0,
        memory_scale_threshold=70.0,
        latency_scale_threshold=200.0,
        cpu_vertical_threshold=75.0,
        memory_vertical_threshold=75.0,
        workload_change_threshold=0.20,
    ):
        self.cpu_scale_threshold = cpu_scale_threshold
        self.memory_scale_threshold = memory_scale_threshold
        self.latency_scale_threshold = latency_scale_threshold

        self.cpu_vertical_threshold = cpu_vertical_threshold
        self.memory_vertical_threshold = memory_vertical_threshold

        self.workload_change_threshold = workload_change_threshold

    def reset(self):
        """
        Reset controller state.

        The current controller is stateless, so there is
        nothing to reset.
        """
        pass

    def select_mode(self, state, info):
        """
        Select the scaling mode based on current system conditions.

        Returns:
            "none"
            "horizontal"
            "vertical"
            "both"
        """

        del state

        cpu = info["cpu_utilization"]
        memory = info["memory_utilization"]
        latency = info["latency"]

        current_workload = info["request_rate"]

        previous_workload = info.get(
            "previous_request_rate",
            current_workload,
        )

        # --------------------------------------------------
        # Workload change
        # --------------------------------------------------

        if previous_workload <= 0:
            workload_change = 0.0
        else:
            workload_change = (
                current_workload - previous_workload
            ) / previous_workload

        # --------------------------------------------------
        # Horizontal scaling pressure
        # --------------------------------------------------

        horizontal_pressure = (
            cpu >= self.cpu_scale_threshold
            or memory >= self.memory_scale_threshold
            or latency >= self.latency_scale_threshold
            or workload_change >= self.workload_change_threshold
        )

        # --------------------------------------------------
        # Vertical scaling pressure
        #
        # Both CPU and memory need to be high.
        # This prevents a single high resource metric from
        # automatically triggering vertical scaling.
        # --------------------------------------------------

        vertical_pressure = (
            cpu >= self.cpu_vertical_threshold
            and memory >= self.memory_vertical_threshold
        )

        # --------------------------------------------------
        # Select mode
        # --------------------------------------------------

        if horizontal_pressure and vertical_pressure:
            return self.MODE_BOTH

        if vertical_pressure:
            return self.MODE_VERTICAL

        if horizontal_pressure:
            return self.MODE_HORIZONTAL

        return self.MODE_NONE