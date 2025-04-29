from abc import ABC, abstractmethod
from typing import TypeAlias

from qiskit import QuantumCircuit
from qiskit.providers import Backend

Assignment: TypeAlias = tuple[QuantumCircuit, Backend]


class BaseEstimator(ABC):
    """
    Base class for estimating job execution time
    """

    @abstractmethod
    def estimate_execution_time(
        self,
        circuits: list[QuantumCircuit],
        backend: Backend,
        **kwargs,
    ) -> float:
        """
        Estimate the execution time of a quantum job on a specified backend
        :param circuits: Circuits in the quantum job
        :param backend: Backend to be executed on
        :param kwargs: Additional arguments, like the run configuration
        :return: Estimated execution time
        """
        ...