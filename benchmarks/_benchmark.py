import abc
from collections import Counter
from typing import List, Sequence, Union

from qiskit.circuit import QuantumCircuit


class Benchmark(abc.ABC):
    @abc.abstractmethod
    def circuit(self) -> Union[QuantumCircuit, Sequence[QuantumCircuit]]:
        """Returns the quantum circuit corresponding to the current benchmark parameters."""

    @abc.abstractmethod
    def score(self, counts: Union[Counter, List[Counter]]) -> float:
        """Returns a normalized [0,1] score reflecting device performance."""
