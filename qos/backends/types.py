from abc import ABC, abstractmethod
from typing import Any, Dict, List
import sys
from qiskit import transpile
from qiskit.providers.fake_provider import *


class Backend(ABC):
    @abstractmethod
    def run(self, args: Dict[str, Any]) -> int:
        pass


# A qernel class should be the counter part of a qernel entry on the Quantum circuit database
class QPU(ABC):
    id: int
    name: str
    provider: str
    backend: str
    shots: int
    alias: str
    args: Dict[str, Any]
    local_queue: List[tuple]  # The local queue will store a tuple of estimated execution time and the time when the circuit was submitted

    def __init__(self) -> None:
        self.name = ""
        self.id = -1
        self.args = {}

    def __str__(self) -> str:
        return (
            "QPU\n id: \t"
            + str(self.id)
            + "\n name: \t"
            + self.name
            + "\n args: \t"
            + str(self.args)
            + "\n"
        )

    def transpile(self, circuit, opt_level=int) -> int:

        if self.provider == "ibm":
            backend = eval(self.name)()
            return transpile(circuit, backend=backend, optimization_level=opt_level)


class Simulator(Backend):
    name: str
    pass
