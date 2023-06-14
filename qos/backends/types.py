from abc import ABC, abstractmethod
from typing import Any, Dict, List
import sys
from threading import Thread, Lock, Semaphore


class Backend(ABC):
    @abstractmethod
    def run(self, args: Dict[str, Any]) -> int:
        pass


# A job class should be the counter part of a job entry on the Quantum circuit database
class QPU(ABC):
    id: int
    name: str
    provider: str
    backend: str
    shots: int
    args: Dict[str, Any]

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


class Simulator(Backend):
    name: str
    pass
