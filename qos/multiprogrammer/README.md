# Multiprogrammer Module

The `multiprogrammer.py` module is part of the QOS framework and provides functionality for scheduling and bundling quantum programs (Qernels) to optimize the utilization of quantum processing units (QPUs). It implements policies for multiprogramming, resource allocation, and re-evaluation of quantum workloads.

## Features

- **Effective Utilization**:
  - Computes spatial and temporal utilization of QPUs.
  - Evaluates effective utilization for pairs of Qernels.

- **Matching and Scoring**:
  - Calculates matching scores between Qernels based on metrics like utilization, entanglement, measurement, and parallelism.
  - Filters and sorts Qernel pairs based on utilization and matching scores.


- **Multiprogramming Policies**:
  - **Restrict Policy**: Merges Qernels only if their layouts do not overlap and share the same QPU.
  - **Re-evaluation Policy**: Bundles Qernels and re-runs the estimator to evaluate the new configuration. This will create a new layout for the bundled Qernel.


## Key Methods

- `spatial_utilization(q1, q2, backend)`: Computes spatial utilization for two Qernels on a backend.
- `effective_utilization(q1, q2, backend)`: Calculates the total effective utilization of a QPU.
- `get_matching_score(q1, q2, backend)`: Computes a matching score for two Qernels.
- `re_evaluation_policy(qernel1, qernel2)`: Bundles two Qernels and re-runs the estimator.
- `restrict_policy(new_qernels, error_limit)`: Implements the restrict policy for merging Qernels.
