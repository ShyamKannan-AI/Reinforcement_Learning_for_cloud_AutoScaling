class ThresholdScaler:
    """
    Traditional rule-based horizontal autoscaler.

    The scaler changes the number of replicas based on
    CPU utilization, memory utilization, and latency.

    Actions:
        0 -> Maintain
        1 -> Scale up
        2 -> Scale down
    """

    MAINTAIN = 0
    SCALE_UP = 1
    SCALE_DOWN = 2

    def __init__(
        self,
        scale_up_cpu_threshold=70.0,
        scale_down_cpu_threshold=30.0,
        scale_up_memory_threshold=70.0,
        scale_down_memory_threshold=30.0,
        scale_up_latency_threshold=200.0,
        scale_down_latency_threshold=100.0,
        cooldown_steps=5,
    ):
        self.scale_up_cpu_threshold = scale_up_cpu_threshold
        self.scale_down_cpu_threshold = scale_down_cpu_threshold

        self.scale_up_memory_threshold = scale_up_memory_threshold
        self.scale_down_memory_threshold = scale_down_memory_threshold

        self.scale_up_latency_threshold = scale_up_latency_threshold
        self.scale_down_latency_threshold = scale_down_latency_threshold

        self.cooldown_steps = cooldown_steps

        self.steps_since_scaling = cooldown_steps

    def reset(self):
        """
        Reset the scaler state before a new episode.
        """

        self.steps_since_scaling = self.cooldown_steps

    def select_action(self, info):
        """
        Select a horizontal scaling action from current metrics.

        Scale up if:
            - CPU is above its upper threshold, OR
            - memory is above its upper threshold, OR
            - latency exceeds the SLA threshold.

        Scale down only if:
            - CPU is below its lower threshold,
            - memory is below its lower threshold, AND
            - latency is comfortably below the SLA.

        A cooldown period prevents repeated scaling actions.
        """

        if self.steps_since_scaling < self.cooldown_steps:
            self.steps_since_scaling += 1
            return self.MAINTAIN

        cpu_utilization = info["cpu_utilization"]
        memory_utilization = info["memory_utilization"]
        latency = info["latency"]

        scale_up = (
            cpu_utilization >= self.scale_up_cpu_threshold
            or memory_utilization >= self.scale_up_memory_threshold
            or latency >= self.scale_up_latency_threshold
        )

        scale_down = (
            cpu_utilization <= self.scale_down_cpu_threshold
            and memory_utilization <= self.scale_down_memory_threshold
            and latency <= self.scale_down_latency_threshold
        )

        if scale_up:
            self.steps_since_scaling = 0
            return self.SCALE_UP

        if scale_down:
            self.steps_since_scaling = 0
            return self.SCALE_DOWN

        self.steps_since_scaling += 1
        return self.MAINTAIN