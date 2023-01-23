from collections import Counter

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector


def _get_ideal_counts(circuit: QuantumCircuit) -> Counter:
    ideal_counts = {}
    sv = Statevector.from_label("0" * circuit.num_qubits)
    circuit_no_meas = circuit.remove_final_measurements(inplace=False)
    sv.evolve(circuit_no_meas)

    for i, amplitude in enumerate(sv):
        bitstring = f"{i:>0{circuit.num_qubits}b}"
        probability = np.abs(amplitude) ** 2
        ideal_counts[bitstring] = probability
    return Counter(ideal_counts)
