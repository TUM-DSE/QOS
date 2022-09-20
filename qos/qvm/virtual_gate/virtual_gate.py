from abc import ABC, abstractmethod
import itertools
from typing import List, Optional, Type
from qiskit.circuit import QuantumCircuit, Instruction, Barrier

from qos.qvm.prob import ProbDistribution


class VirtualBinaryGate(Barrier, ABC):
    def __init__(self, params: Optional[List] = None):
        super().__init__(2)
        if params is None:
            params = []
        self._params = params

    @abstractmethod
    def configure(self) -> List[QuantumCircuit]:
        pass

    @abstractmethod
    def knit(self, results: List[ProbDistribution]) -> ProbDistribution:
        pass

    def configuration(self, config_id: int) -> QuantumCircuit:
        return self.configure()[config_id]
