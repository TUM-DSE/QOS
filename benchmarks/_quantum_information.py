from collections import Counter
from typing import List, Union

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity

from ._benchmark import Benchmark


class GHZBenchmark(Benchmark):
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        qc.h(0)
        for i in range(1, self.num_qubits):
            qc.cnot(i - 1, i)

        qc.measure_all()

        return qc

    def score(self, counts: Union[Counter, List[Counter]]) -> float:
        """
        Compute the Hellinger fidelity between the experimental and ideal
        results, i.e., 50% probabilty of measuring the all-zero state and 50%
        probability of measuring the all-one state.
        """
        assert isinstance(counts, Counter)

        # Create an equal weighted distribution between the all-0 and all-1 states
        ideal_dist = {b * self.num_qubits: 0.5 for b in ["0", "1"]}
        total_shots = sum(counts.values())
        device_dist = {bitstr: count / total_shots for bitstr, count in counts.items()}
        return hellinger_fidelity(ideal_dist, device_dist)


class MerminBellBenchmark(Benchmark):
    # TODO
    pass
