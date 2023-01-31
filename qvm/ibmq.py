from typing import Any, Dict, List, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.providers.ibmq import IBMQBackend

from vqc.device.device import Device
from vqc.prob import ProbDistribution


class IBMQDevice(Device):
    def __init__(
        self, backend: IBMQBackend, transpiler_options: Optional[Dict[str, Any]] = None
    ):
        self.backend = backend
        self.transpiler_options = transpiler_options

    def run(self, circuits: List[QuantumCircuit], shots: int) -> List[ProbDistribution]:
        if len(circuits) == 0:
            return []
        circuits = transpile(circuits, self.backend)
        if len(circuits) == 1:
            return [
                ProbDistribution.from_counts(
                    self.backend.run(circuits[0], shots=shots).result().get_counts()
                )
            ]
        return [
            ProbDistribution.from_counts(
                self.backend.run(circ, shots=shots).result().get_counts()
            )
            for circ in circuits
        ]
