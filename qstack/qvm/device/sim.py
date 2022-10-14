from typing import List

from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator

from vqc.prob import ProbDistribution

from .device import Device


class SimDevice(Device):
    def run(self, circuits: List[QuantumCircuit], shots: int) -> List[ProbDistribution]:
        backend = AerSimulator()
        if len(circuits) == 0:
            return []

        circuits = transpile(circuits, backend)
        if len(circuits) == 1:
            return [
                ProbDistribution.from_counts(
                    backend.run(circuits[0], shots=shots).result().get_counts()
                )
            ]
        return [
            ProbDistribution.from_counts(
                backend.run(circ, shots=shots).result().get_counts()
            )
            for circ in circuits
        ]
