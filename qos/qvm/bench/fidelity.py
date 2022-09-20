from typing import Dict
from qiskit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity
from qiskit.providers.aer import StatevectorSimulator


def perfect_counts(original_circuit: QuantumCircuit) -> Dict[str, int]:
    cnt = (
        StatevectorSimulator().run(original_circuit, shots=500000).result().get_counts()
    )
    return {k.replace(" ", ""): v for k, v in cnt.items()}


def fidelity(orginal_circuit: QuantumCircuit, noisy_counts: Dict[str, int]) -> float:
    return hellinger_fidelity(perfect_counts(orginal_circuit), noisy_counts)
