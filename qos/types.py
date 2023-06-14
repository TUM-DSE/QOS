from abc import ABC, abstractmethod
from typing import Any, Dict, List
import sys
from threading import Thread, Lock, Semaphore
from qos.backends.types import QPU


# This should contain any kind of circuit, qiskit circuit or circ, etc
class QCircuit(ABC):
    id: int
    type: str
    args: Dict[str, Any]
    _circuit: str

    def __init__(self) -> None:
        self.args = {}
        self._circuit = None


# A job class should be the counter part of a job entry on the Quantum circuit database
class Job(ABC):
    id: int
    status: str
    qpu: QPU
    circuits: List[QCircuit]
    args: Dict[str, Any]

    def __init__(self) -> None:
        self.args = {}
        self.subjobs: List[Job] = []


class Engine(ABC):
    """Generic interface for implementations of QOS Engines"""

    @abstractmethod
    def submit(self, args: Dict[str, Any]) -> int:
        pass


#    @abstractmethod
#    def execute(self, id: int, args: Dict[str, Any]) -> None:
#        pass

#    @abstractmethod
#    def fetch(self, id: int, args: Dict[str, Any]) -> None:
#        pass


class Backend(ABC):
    @abstractmethod
    def run(self, args: Dict[str, Any]) -> int:
        pass


class scheduler_policy(ABC):
    @abstractmethod
    def schedule(self, new_job: Job, kargs: Dict):
        pass
