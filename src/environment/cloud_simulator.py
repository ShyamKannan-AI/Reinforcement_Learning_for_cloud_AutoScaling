import numpy as np


class CloudSimulator:
    """
    Simulates a cloud application for reinforcement-learning-based
    auto-scaling.

    The simulator models:
    - Incoming workload
    - CPU and memory capacity
    - CPU and memory utilization
    - Application latency
    - Infrastructure cost
    - SLA violations
    - Horizontal scaling
    - Vertical scaling
    - Reward calculation
    """

    def __init__(
        self,
        min_replicas=1,
        max_replicas=20,
        min_cpu=1.0,
        max_cpu=8.0,
        min_memory=2.0,
        max_memory=32.0,
        max_steps=500,
        seed=None,
    ):
        # Resource limits
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

        self.min_cpu = min_cpu
        self.max_cpu = max_cpu

        self.min_memory = min_memory
        self.max_memory = max_memory

        # Simulation configuration
        self.max_steps = max_steps

        # Random number generator
        self.rng = np.random.default_rng(seed)

        # Initial resource configuration
        self.initial_replicas = 2
        self.initial_cpu = 2.0
        self.initial_memory = 4.0

        # Workload configuration
        self.base_workload = 100.0

        # SLA configuration
        self.sla_latency = 200.0

        # Simulation state
        self.replicas = self.initial_replicas
        self.cpu_per_replica = self.initial_cpu
        self.memory_per_replica = self.initial_memory

        self.request_rate = self.base_workload
        self.previous_request_rate = self.base_workload

        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.latency = 0.0
        self.cost = 0.0

        self.sla_violation = 0

        self.step_count = 0

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self):
        """
        Reset the simulator to its initial state.

        Returns:
            dict: Initial simulation state.
        """

        self.replicas = self.initial_replicas
        self.cpu_per_replica = self.initial_cpu
        self.memory_per_replica = self.initial_memory

        self.request_rate = self.base_workload
        self.previous_request_rate = self.base_workload

        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.latency = 0.0
        self.cost = 0.0

        self.sla_violation = 0

        self.step_count = 0

        # Generate the first workload
        self.request_rate = self._generate_workload()

        # Calculate initial metrics
        self._update_metrics()

        return self.get_state()

    # ------------------------------------------------------------------
    # WORKLOAD
    # ------------------------------------------------------------------

    def _generate_workload(self):
        """
        Generate workload for the current timestep.

        Workload consists of:
        - baseline traffic
        - periodic traffic
        - random noise
        - occasional traffic spikes

        Returns:
            float: Request rate.
        """

        periodic_component = 50.0 * np.sin(self.step_count / 20.0)

        noise = self.rng.normal(0.0, 15.0)

        workload = (
            self.base_workload
            + periodic_component
            + noise
        )

        # Occasional sudden workload spike
        if self.rng.random() < 0.05:
            spike = self.rng.uniform(100.0, 400.0)
            workload += spike

        return max(0.0, workload)

    # ------------------------------------------------------------------
    # CAPACITY
    # ------------------------------------------------------------------

    def _calculate_cpu_capacity(self):
        """
        Calculate total CPU capacity.

        Returns:
            float: Total CPU capacity.
        """

        return self.replicas * self.cpu_per_replica * 50.0

    def _calculate_memory_capacity(self):
        """
        Calculate total memory capacity.

        Returns:
            float: Total memory capacity.
        """

        return self.replicas * self.memory_per_replica * 25.0

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

    def _update_metrics(self):
        """
        Update CPU utilization, memory utilization,
        latency, cost, and SLA status.
        """

        cpu_capacity = self._calculate_cpu_capacity()
        memory_capacity = self._calculate_memory_capacity()

        # Prevent division by zero
        cpu_capacity = max(cpu_capacity, 1e-6)
        memory_capacity = max(memory_capacity, 1e-6)

        self.cpu_utilization = (
            self.request_rate / cpu_capacity
        ) * 100.0

        self.memory_utilization = (
            self.request_rate / memory_capacity
        ) * 100.0

        # Calculate resource pressure
        cpu_pressure = max(
            0.0,
            (self.cpu_utilization - 70.0) / 30.0
        )

        memory_pressure = max(
            0.0,
            (self.memory_utilization - 70.0) / 30.0
        )

        pressure = max(cpu_pressure, memory_pressure)

        # Base latency
        base_latency = 50.0

        # Latency increases when resource pressure becomes high
        self.latency = base_latency + (
            pressure * 450.0
        )

        # Infrastructure cost
        replica_cost = self.replicas * 1.0
        cpu_cost = (
            self.replicas
            * self.cpu_per_replica
            * 0.5
        )
        memory_cost = (
            self.replicas
            * self.memory_per_replica
            * 0.1
        )

        self.cost = (
            replica_cost
            + cpu_cost
            + memory_cost
        )

        # SLA violation
        self.sla_violation = int(
            self.latency > self.sla_latency
        )

    # ------------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------------

    def get_state(self):
        """
        Construct the normalized state observed by the RL system.

        State:
        [
            workload,
            CPU utilization,
            memory utilization,
            latency,
            replicas,
            CPU per replica,
            memory per replica,
            workload change
        ]

        Returns:
            np.ndarray: Normalized state vector.
        """

        workload_change = (
            self.request_rate
            - self.previous_request_rate
        )

        state = np.array(
            [
                self.request_rate / 1000.0,
                self.cpu_utilization / 100.0,
                self.memory_utilization / 100.0,
                self.latency / 500.0,
                self.replicas / self.max_replicas,
                self.cpu_per_replica / self.max_cpu,
                self.memory_per_replica / self.max_memory,
                workload_change / 500.0,
            ],
            dtype=np.float32,
        )

        return state

    # ------------------------------------------------------------------
    # HORIZONTAL SCALING
    # ------------------------------------------------------------------

    def scale_horizontal(self, direction):
        """
        Change the number of replicas.

        Args:
            direction:
                +1 -> add one replica
                -1 -> remove one replica
                 0 -> no change
        """

        if direction > 0:
            self.replicas = min(
                self.replicas + 1,
                self.max_replicas,
            )

        elif direction < 0:
            self.replicas = max(
                self.replicas - 1,
                self.min_replicas,
            )

    # ------------------------------------------------------------------
    # VERTICAL SCALING
    # ------------------------------------------------------------------

    def scale_cpu(self, direction):
        """
        Increase or decrease CPU allocated to each replica.

        Args:
            direction:
                +1 -> increase CPU
                -1 -> decrease CPU
                 0 -> no change
        """

        cpu_step = 1.0

        if direction > 0:
            self.cpu_per_replica = min(
                self.cpu_per_replica + cpu_step,
                self.max_cpu,
            )

        elif direction < 0:
            self.cpu_per_replica = max(
                self.cpu_per_replica - cpu_step,
                self.min_cpu,
            )

    def scale_memory(self, direction):
        """
        Increase or decrease memory allocated to each replica.

        Args:
            direction:
                +1 -> increase memory
                -1 -> decrease memory
                 0 -> no change
        """

        memory_step = 2.0

        if direction > 0:
            self.memory_per_replica = min(
                self.memory_per_replica + memory_step,
                self.max_memory,
            )

        elif direction < 0:
            self.memory_per_replica = max(
                self.memory_per_replica - memory_step,
                self.min_memory,
            )

    # ------------------------------------------------------------------
    # REWARD
    # ------------------------------------------------------------------

    def calculate_reward(self):
        """
        Calculate the reward for the current state.

        The reward balances:
        - Infrastructure cost
        - Latency
        - SLA violations
        - Resource utilization

        Returns:
            float: Reward value.
        """

        # Cost penalty
        cost_penalty = 0.5 * self.cost

        # Latency penalty
        latency_penalty = 0.01 * self.latency

        # Strong penalty for SLA violations
        sla_penalty = 10.0 * self.sla_violation

        # Reward reasonable resource utilization
        average_utilization = (
            self.cpu_utilization
            + self.memory_utilization
        ) / 2.0

        utilization_reward = 0.0

        if 40.0 <= average_utilization <= 80.0:
            utilization_reward = 2.0

        # Penalize severe under-utilization
        underutilization_penalty = 0.0

        if average_utilization < 20.0:
            underutilization_penalty = 1.0

        reward = (
            -cost_penalty
            -latency_penalty
            -sla_penalty
            +utilization_reward
            -underutilization_penalty
        )

        return float(reward)

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(self):
        """
        Advance the simulation by one timestep.

        The simulator itself does not decide the scaling action.
        The controller/agent will modify resources before this
        method is called.

        Returns:
            tuple:
                state
                reward
                terminated
                info
        """

        self.step_count += 1

        # Save current workload for calculating workload change
        self.previous_request_rate = self.request_rate

        # Generate new workload
        self.request_rate = self._generate_workload()

        # Calculate new system metrics
        self._update_metrics()

        # Calculate reward
        reward = self.calculate_reward()

        # Check termination
        terminated = self.step_count >= self.max_steps

        state = self.get_state()

        info = {
            "request_rate": self.request_rate,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "latency": self.latency,
            "cost": self.cost,
            "sla_violation": self.sla_violation,
            "replicas": self.replicas,
            "cpu_per_replica": self.cpu_per_replica,
            "memory_per_replica": self.memory_per_replica,
        }

        return state, reward, terminated, info