from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator

from qvm.quasi_distr import QuasiDistr


def calculate_fidelity(circuit: QuantumCircuit, noisy_result: QuasiDistr) -> float:
    ideal_result = QuasiDistr.from_counts(
        AerSimulator().run(circuit, shots=20000).result().get_counts()
    )
    print(ideal_result)
    return hellinger_fidelity(ideal_result, noisy_result)
