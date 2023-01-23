from abc import ABC, abstractmethod
import itertools
from typing import List, Optional, Type
from qiskit.circuit import QuantumCircuit, Instruction, Barrier

from vqc.prob import ProbDistribution


class VirtualBinaryGate(Barrier, ABC):

    _ids = itertools.count(0)

    def __init__(self, params: Optional[List] = None):
        super().__init__(2)
        if params is None:
            params = []
        self.id = next(self._ids)
        self._params = params

    def __eq__(self, other):
        return super().__eq__(other) and self.id == other.id

    def __hash__(self):
        return self.id

    @abstractmethod
    def configure(self) -> List[QuantumCircuit]:
        pass

    @abstractmethod
    def knit(self, results: List[ProbDistribution]) -> ProbDistribution:
        pass

    def configuration(self, config_id: int) -> QuantumCircuit:
        return self.configure()[config_id]
