# QOS Scheduler

The `scheduler` module is a core component of the QOS framework, responsible for scheduling quantum circuits across multiple QPUs. It optimizes fidelity, waiting times, and QPU utilization using advanced scheduling policies, including formula-based and genetic algorithm-based approaches.

## Module Structure

### 1. **Execution Time Estimation**
- **Description**: A separate component that estimates the execution time of a Qernel.
- **Key Features**:
  - Calculates the execution time by analyzing the longest-duration gate chain in the Qernel's Directed Acyclic Graph (DAG).
  - Provides accurate estimates to optimize scheduling decisions.

---

### 2. **`scheduler.py`**
- **Description**: Implements the basic scheduling logic, including execution time estimation and scheduling policies.
- **Key Features**:
  - **Formula-Based Policy**:
    - Optimizes for conflicting objectives (e.g., fidelity vs. waiting time) using a scoring formula:
      ```
      Score = c * (f2 - f1) / f1 + (1 - c) * (t2 - t1) / t1 + β * (u2 - u1) / u1
      ```
      - `f`: Fidelity of the circuit on the QPU.
      - `t`: Waiting time for the QPU.
      - `u`: Utilization of the QPU.
      - `c`: Weighting factor for fidelity vs. waiting time (default: `c = 0.5`).
      - `β`: Weighting factor for utilization (default: `β = 0.5`).
    - Balances fidelity, waiting time, and utilization to select the best schedule.

---

### 3. **`multi_objective_scheduler.py`**
- **Description**: Implements a multi-objective optimization approach using genetic algorithms.
- **Key Features**:
  - **Genetic Algorithm Policy**:
    - Uses the NSGA-II genetic algorithm to optimize for conflicting objectives (fidelity vs. waiting time).
    - Generates a Pareto front of possible schedules, each achieving a different trade-off between fidelity and waiting time.
    - Selects the best schedule from the Pareto front using the scoring formula described in the formula-based policy.
