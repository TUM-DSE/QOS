# QVM Project

The `qvm` project is a framework for optimizing quantum circuits with circuit cutting, qubit reuse, and gate dependency reduction. It provides tools for breaking down large quantum circuits into smaller fragments that can be executed on multiple QPUs and later recombined (knitting) using classical post-processing.

## Project Structure

The project is organized into the following components:

### 1. **Compiler**
The `compiler` module contains passes for optimizing and transforming quantum circuits. These include:

- **Qubit Reuser** (`distr_transpiler/qubit_reuser.py`):
  - Implements qubit reuse by replacing `measure-reset` operations with dynamic conditional gates.
  - Reduces the number of qubits required for execution.

- **Gate and Wire Cutters** (`virtualization/gate_decomp.py`, `virtualization/wire_decomp.py`):
  - Implements techniques for cutting quantum circuits at specific gates or wires to split them into smaller fragments.
  - Enables distributed execution of large circuits.

- **Dependency Reducer** (`virtualization/reduce_deps.py`):
  - Reduces dependencies between qubits in a circuit to improve parallelism and reduce communication overhead.

- **Intermediate Representation (IR)** (`dag.py`):
    - The `dag` module provides a Directed Acyclic Graph (DAG)-based intermediate representation for quantum circuits. This is used internally by the compiler passes for efficient manipulation of circuits.

- **Types** (`types.py`):
    - The `types` module defines core data structures and interfaces used throughout the project, such as the `DistributedTranspilerPass` base class.

### 2. **Quasi-Distribution** (`quasi_distr.py`)
This module provides tools for classical post-processing of results from distributed quantum circuits. It includes methods for "knitting" the results of circuit fragments back together into a single quasi-probability distribution.

### 3. **Virtual Circuit** (`virtual_circuit.py`)
The `VirtualCircuit` module is a wrapper around Qiskit's `QuantumCircuit`. It introduces the concept of virtual gates, which act as placeholders for operations that are instantiated later during execution.

### 4. **Virtual Gates** (`virtual_gates.py`)
This module defines virtual gate placeholders that represent operations split across circuit fragments. Key features include:

- **WireCut**:
  - Represents a wire cut in the circuit.
  - Acts as a barrier for splitting circuits.

- **VirtualBinaryGate**:
  - Abstract base class for virtual two-qubit gates.
  - Supports instantiation of multiple circuit fragments and post-processing to combine results.

- **Specific Virtual Gates**:
  - `VirtualMove`: Implements virtual move operations.
  - `VirtualCX`, `VirtualCY`, `VirtualCZ`: Represent virtual controlled gates.
  - `VirtualRZZ`, `VirtualCPhase`: Represent parameterized virtual gates.

### 5. **Core Features**
The `qvm` project implements the following core features:
- **Circuit Cutting**:
  - Splits large circuits into smaller fragments using gate and wire cutting techniques.
  - Enables distributed execution on hardware with limited qubits.

- **Qubit Reuse**:
  - Reduces the number of qubits required by reusing qubits dynamically.
  - Implements dynamic conditional gates to replace `measure-reset` operations.

- **Classical Post-Processing**:
  - Combines results from circuit fragments using quasi-probability distributions.
  - Supports efficient recombination of distributed results.

