# Adaptive Multi-Agent Reinforcement Learning for Joint Horizontal and Vertical Auto-Scaling of Cloud Resources

## Overview

Cloud applications experience dynamic workloads that can cause increased response latency, Service Level Agreement (SLA) violations, and inefficient resource utilization. Traditional threshold-based auto-scaling approaches often struggle to respond effectively to fluctuating workloads.

This project presents an **Adaptive Multi-Agent Reinforcement Learning Framework** that coordinates horizontal and vertical auto-scaling using two independent reinforcement learning agents:

- **Deep Q-Network (DQN):** Handles horizontal scaling by increasing, decreasing, or maintaining the number of application replicas.
- **Proximal Policy Optimization (PPO):** Handles vertical scaling by adjusting CPU and memory allocation per replica.
- **Adaptive Scaling Controller:** Selects the appropriate scaling mode based on workload changes, CPU utilization, memory utilization, and response latency.

Both agents operate within a shared custom Python-based cloud simulation environment and receive feedback through a common reward function.

The framework is evaluated against threshold-based scaling, DQN-based horizontal scaling, and PPO-based vertical scaling.

## Key Features

- Multi-agent reinforcement learning for cloud resource management.
- DQN-based horizontal auto-scaling.
- PPO-based vertical auto-scaling.
- Adaptive controller for coordinating scaling decisions.
- Custom Python-based cloud simulation environment.
- Synthetic workload generation with periodic variations, random noise, and workload spikes.
- Shared reward function considering resource cost, latency, SLA violations, and utilization.
- Performance evaluation using multiple metrics.
- Unit testing and evaluation components.

## System Architecture

The framework consists of the following major components:

1. Workload Generator
2. Cloud Simulation Environment
3. State Representation Module
4. Adaptive Scaling Controller
5. DQN Horizontal Scaling Agent
6. PPO Vertical Scaling Agent
7. Reward and Evaluation Module

### Architecture Diagram

![Adaptive Multi-Agent RL Cloud Auto-Scaling Architecture](framework_architecture.png)

The workload generator produces changing request rates. The cloud simulator calculates resource utilization, latency, and simulated resource costs. These observations are provided to the scaling controller and the reinforcement learning agents.

The DQN agent manages the number of replicas, while the PPO agent manages CPU and memory allocation. The selected actions are applied to the simulator, which calculates the next state and shared reward.

## Reinforcement Learning Agents

### 1. DQN – Horizontal Auto-Scaling

The DQN agent manages the number of application replicas.

**Action space:**

| Action | Description |
|---|---|
| 0 | Maintain replica count |
| 1 | Increase replicas by one |
| 2 | Decrease replicas by one |

The replica count is constrained between 1 and 20.

**Network architecture:**
- Input dimension: 8 state features.
- Two hidden layers with 128 neurons each.
- ReLU activation.
- Three output action values.
- Target network.
- Experience replay buffer.
- Epsilon-greedy action selection.
- Adam optimizer.
- Smooth L1 loss.

### 2. PPO – Vertical Auto-Scaling

The PPO agent adjusts CPU and memory allocation per replica.

**Action space:**

| Action | Description |
|---|---|
| 0 | Maintain allocation |
| 1 | Increase CPU by 1.0 |
| 2 | Decrease CPU by 1.0 |
| 3 | Increase memory by 2.0 |
| 4 | Decrease memory by 2.0 |

**Resource allocation constraints:**

- CPU allocation: 1.0–8.0 units.
- Memory allocation: 2.0–32.0 units.

The PPO implementation uses:
- Separate actor and critic networks.
- Rollout buffer.
- Generalized Advantage Estimation (GAE).
- Clipped policy optimization objective.
- Rollout length of 128 steps.

### 3. Adaptive Scaling Controller

The deterministic controller determines which scaling mode should be activated.

The available modes are:

- No scaling.
- Horizontal scaling.
- Vertical scaling.
- Combined horizontal and vertical scaling.

The controller considers CPU utilization, memory utilization, response latency, and relative workload changes.

The horizontal pressure conditions use CPU and memory thresholds of 70%, latency of 200 ms, and a relative workload increase of 20%. Vertical scaling is activated when both CPU and memory utilization reach at least 75%.

**Implementation note:** With the default thresholds, the vertical-only mode is unreachable because satisfying the vertical condition also satisfies the horizontal CPU and memory conditions.

## Cloud Simulation Environment

The project uses a custom Python-based cloud simulator rather than a real Kubernetes cluster or public cloud deployment.

The simulator models:

- Application replicas.
- CPU and memory allocation.
- Workload fluctuations.
- CPU and memory utilization.
- Response latency.
- Simulated resource cost.
- SLA violations.

### State Representation

The agents receive an eight-dimensional state vector containing:

1. Request rate.
2. CPU utilization.
3. Memory utilization.
4. Response latency.
5. Replica count.
6. CPU allocation.
7. Memory allocation.
8. Change in request rate.

### Reward Function

Both agents receive a shared reward that considers:

- Resource cost.
- Response latency.
- SLA violations.
- Moderate resource utilization.
- Underutilization.

This encourages the agents to balance application performance and resource consumption.

## Experimental Configuration

| Parameter | Configuration |
|---|---|
| Simulation environment | Custom Python-based simulator |
| Workload | Synthetic dynamic workload |
| Initial replicas | 2 |
| Minimum replicas | 1 |
| Maximum replicas | 20 |
| Initial CPU allocation | 2.0 units |
| Initial memory allocation | 4.0 units |
| CPU allocation range | 1.0–8.0 units |
| Memory allocation range | 2.0–32.0 units |
| Training episodes | 100 |
| Maximum steps per episode | 500 |
| PPO rollout length | 128 steps |
| Evaluation episodes | 10 |
| SLA latency threshold | 200 ms |

## Experimental Results

The proposed adaptive framework was evaluated against three other strategies:

1. Threshold-based auto-scaling.
2. DQN-based horizontal scaling.
3. PPO-based vertical scaling.

The following table presents the recorded evaluation results.

| Evaluation Metric | Threshold | DQN | PPO | Adaptive |
|---|---:|---:|---:|---:|
| Average reward per step | -4.1973 | -5.0820 | **-3.9302** | -4.0942 |
| Average latency (ms) | 133.72 | 258.92 | 99.61 | **87.10** |
| P95 latency (ms) | 407.24 | 1028.08 | 215.59 | **153.53** |
| SLA violation rate (%) | 6.96 | 20.74 | 4.98 | **4.62** |
| Average replicas | 2.55 | 1.57 | 2.00 | 1.09 |
| Replica-count changes/episode | 46.4 | 48.1 | 0.0 | 8.2 |
| Mean episode total cost | 3059.76 | 1876.56 | 3022.10 | 2983.79 |

### Results Discussion

The adaptive framework achieved the lowest average latency of 87.10 ms and P95 latency of 153.53 ms among the evaluated strategies. It also recorded the lowest SLA violation rate of 4.62%.

The adaptive approach reduced average latency compared with threshold-based scaling and DQN-based horizontal scaling in the evaluated simulation.

The PPO strategy achieved the highest average reward per step of -3.9302, while the adaptive strategy achieved -4.0942.

The adaptive framework recorded a mean episode cost of 2983.79. This was lower than threshold-based scaling but higher than DQN and PPO in the reported evaluation. Therefore, the results demonstrate a trade-off between simulated resource cost and service performance rather than a universal cost advantage.

**Important:** These results are based on ten evaluation episodes using synthetic workloads in a custom simulator. They should not be interpreted as production-cloud benchmarks or statistically significant performance improvements.

## Acknowledgements

This project was developed as a research-oriented implementation exploring reinforcement learning techniques for adaptive cloud resource management.


A license has not yet been specified. Add an appropriate open-source license before granting others permission to reuse, modify, or distribute this project.
