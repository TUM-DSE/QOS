from abc import ABC, abstractmethod
from typing import List

from qiskit import QuantumCircuit

from vqc.prob import ProbDistribution


class Device(ABC):
    @abstractmethod
    def run(self, circuits: List[QuantumCircuit], shots: int) -> List[ProbDistribution]:
        pass


