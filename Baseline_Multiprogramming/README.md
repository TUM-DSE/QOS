# Multiprogramming Module

The `multiprogramming.py` module provides tools and algorithms for efficient qubit allocation, scheduling, and analysis of quantum programs. It supports both independent and shared qubit allocation strategies, ensuring optimal utilization of quantum hardware.

## Features

- **Backend Information Retrieval**:
  - `get_all_backend_info`: Extracts detailed information about a quantum backend, including coupling map, qubit properties, and supported features.

- **Qubit Utility Computation**:
  - `compute_qubit_utility`: Calculates the utility of each qubit based on connectivity and error rates.

- **Circuit Analysis**:
  - `compute_CMR`: Computes the Compute-to-Measurement Ratio (CMR) for a quantum circuit.
  - `analyze_programs`: Analyzes quantum programs to compute qubit usage, interactions, and CMR.

- **Qubit Allocation**:
  - `create_sub_graph`: Identifies reliable clusters on the quantum chip.
  - `fair_and_reliable_partition`: Allocates qubits for fair and reliable execution.

- **Scheduling**:
  - `generate_schedule`: Schedules a quantum circuit using Qiskit's ALAP scheduling pass.
  - `independent_qubit_allocation_and_scheduling`: Allocates and schedules qubits independently for each program.
  - `shared_qubit_allocation_and_scheduling`: Allocates and schedules qubits across multiple programs, allowing shared usage.

- **Error Checking**:
  - `shared_scheduling_with_error_check`: Performs shared scheduling and checks for mean error rate warnings.
