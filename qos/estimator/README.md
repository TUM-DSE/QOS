# Estimator Module

The `estimator.py` module is part of the QOS (Quantum Operating System) framework and provides functionality for predicting a Qernel's fidelity across multiple quantum backends. It includes variable methods for determining the best layout for a quantum circuit across QPUs.

## Features

- **Initialization**: Automatically loads available quantum backends and their properties, including gate errors, readout errors, and qubit coherence times (`T1`, `T2`).
- **Layout Optimization**: Provides methods to find the best layout for a quantum circuit on a given backend using cost functions.
- **Cost Functions**: Supports multiple cost functions, including:
  - `trivialConstFunction`: A simple fidelity-based cost function.
  - `accurate_cost_func`: A detailed cost function considering gate errors, readout errors, and idle errors.
  - `regression_cost_function`: A machine-learning-based cost function for predicting circuit fidelity.
- **Execution Time Estimation**: Integrates with the `mapomatic` library to evaluate layouts and estimate execution times.
- **Support for Multiple Backends**: Handles both real and simulated quantum backends, including IBM Quantum backends and fake backends for testing.

## Key Classes and Methods

### `Estimator` Class

The main class in the module, inheriting from `Engine`. It provides the following key methods:

- **Initialization**:
  - `__init__(qpus: list = None, model_path: str = "qos/estimator")`: Initializes the estimator with available backends and loads a machine learning model for regression-based cost estimation.

- **Backend Property Retrieval**:
  - `getMedianReadoutError(backend)`: Returns the median readout error for a backend.
  - `getMedianGateError(backend, gate)`: Returns the median gate error for a specific gate on a backend.
  - `getMedianT1(backend)`: Returns the median `T1` time for a backend.
  - `getMedianT2(backend)`: Returns the median `T2` time for a backend.

- **Layout Optimization**:
  - `best_overall_layoutv2(circuit, backends, successors=True, cost_function=trivialConstFunction)`: Finds the best layout for a circuit across multiple backends using a specified cost function.
  - `best_overall_layoutv3(circuit, backends, successors=True, cost_function=trivialConstFunction)`: An alternative implementation of layout optimization with additional features.

- **Cost Functions**:
  - `trivialConstFunction(circuit, layouts, backend)`: A simple cost function based on gate and readout errors.
  - `accurate_cost_func(circ, layouts, backend)`: A detailed cost function considering gate errors, readout errors, and idle errors.
  - `regression_cost_function(qernel)`: A machine-learning-based cost function for predicting circuit fidelity.

- **Execution**:
  - `run(qernel: Qernel, cost_function=trivialConstFunction)`: Executes the layout optimization process for a given quantum kernel (`Qernel`).

## Usage

### Example: Finding the Best Layout for a Circuit

```python
from qos.estimator.estimator import Estimator
from qos.types.types import Qernel
from qiskit import QuantumCircuit

# Initialize the estimator
estimator = Estimator()

# Create a quantum circuit
qc = QuantumCircuit(5)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

qernel = Qernel(qc)

# Find the best layout
best_layouts = estimator.run(qc,successors=True)
print("Best Layout:", best_layouts) #returns a sorted list of QPUs and their fidelity predictions.