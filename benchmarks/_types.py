from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Sequence, Union, Dict, Optional

from vqc.prob import ProbDistribution
from vqc.device.device import Device

from qiskit.circuit import QuantumCircuit, transpile
from qiskit.providers.ibmq import IBMQBackend
from typing import List

from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator

from vqc.prob import ProbDistribution

from ..benchmarks.device import Device


class Benchmark(ABC):
    @abstractmethod
    def circuit(self) -> Union[QuantumCircuit, Sequence[QuantumCircuit]]:
        """Returns the quantum circuit corresponding to the current benchmark parameters."""

    @abstractmethod
    def score(self, counts: Union[Counter, List[Counter]]) -> float:
        """Returns a normalized [0,1] score reflecting device performance."""


class Device(ABC):
    @abstractmethod
    def run(self, circuits: List[QuantumCircuit], shots: int) -> List[ProbDistribution]:
        pass


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
