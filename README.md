# QOS: Quantum Operating System

This repository contains the source code for the **Quantum Operating System (QOS)**, as described in the paper [QOS: Quantum Operating System](https://arxiv.org/pdf/2406.19120), to appear in Usenix Symposium on Operating Systems Design and Implementation ([OSDI](https://www.usenix.org/conference/osdi25)) 2025. 

## Key Features of QOS

1. **Qernel Abstraction**:
   - QOS implements diverse mechanisms across the stack, including circuit optimization, error mitigation, and spatio-temporal multiplexing on QPUs.
   - The Qernel abstraction acts as a common denominator for composing the QOS mechanisms in a single stack.

2. **Error mitigation**:
    - To address the low fidelity of current QPUs, the error mitigator composes complementary error mitigation and circuit optimization techniques in a single pipeline.
    -  Supports circuit cutting (wire and gate cutting), qubit freezing, and qubit reuse.

3. **Performance Estimation**:
   - To guide multi-programming and scheduling decisions on heterogeneous and noisy QPUs, QOS implements performance estimation.
   - QOS does not simulate circuits (exponential overheads) to estimate fidelity. Instead, it uses numerical- and  regression-based policies.

4. **Multi-programming**:
   - To increase QPU utilization, QOS co-locates multiple circuits on a single QPU (possibly from different users, i.e., supports multi-tenancy).
   - QOS increases *effective* utilization, i.e., spatial and temporal utilization, by taking into account circuit runtimes.
   - To mitigate destructive interference between circuits, which degrades performance, QOS implements *compatibility functions* that quantify how well any two circuits fit together. 
   - QOS implements *buffer zones* between circuit allocations, to further avoid destructive interference.

5. **Scheduling**:
   - There is an inherent tradeoff between execution quality (quantum fidelity) and latency (i.e., waiting times/JCTs) when scheduling circuits on heterogeneous noisy QPUs.
   - QOS implements multi-objective scheduling policies that strike a balance between these conflicting objectives.

## Project Structure

### `/Baseline_Multiprogramming`
- **Description**: Implements the techniques from the paper ["A Case for Multi-Programming Quantum Computers"](https://dl.acm.org/doi/abs/10.1145/3352460.3358287) by Das et al., MICRO 2019. 

---

### `/data`
- **Description**: Contains Redis database configuration and calibration data from (now deprecated) QPUs.
- **Key Features**:
  - Provides calibration data for benchmarking and evaluation.
  - Includes Redis-based configuration for managing quantum workload metadata.

---

### `/evaluation`
- **Description**: Contains data from QOS's evaluation.
- **Subdirectories**:
  - **`/benchmarks`**:
    - **Description**: State-of-the-art quantum benchmarks, including:
      - **Adder**, **Bernstein-Vazirani (BV)**, **GHZ**, **Hamiltonian Simulation**, **QAOA**, **QSM**, **TwoLocal**, **VQE**, and **WState**.
    - **Features**:
      - Configurable circuit structures, e.g., QAOA with regular, power-law, and barbell graphs for Max-Cut problems.
      - `circuits.py`: A utility for generating new benchmarks or fetching existing ones.
    - **Example**:
      ```python
      from circuits import qaoa
      circuit = qaoa(nx.barbell_graph(5, 0))
      ```
  - **`data`**:
    - **Description**: Contains evaluation data, though not all datasets are up-to-date with the paper.
  - **`/plots`**:
    - **Description**: Includes the plots used in the QOS paper presented at the conference.

---

### `/FrozenQubits`
- **Description**: The [source code](https://zenodo.org/records/7278398) of the ASPLOS '23 paper ["FrozenQubits: Boosting Fidelity of QAOA by Skipping Hotspot Nodes"](https://dl.acm.org/doi/abs/10.1145/3575693.3575741) by Ayanzadeh et al. There are minor adapations to make it work with QOS.

---

### `/qos`
- **Description**: The main source code for the QOS system. Please check the internal READMEs.
---

### `/qvm`
- **Description**: The main source code for the **Quantum Virtual Machine (QVM)** project which implements circuit cutting and qubit reuse, which are leveraged by QOS's error mitigator. Please check the internal READMEs.
---

