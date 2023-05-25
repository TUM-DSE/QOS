from abc import ABC, abstractmethod
from typing import Any, Dict, List
import sys
from threading import Thread, Lock, Semaphore


class Backend(ABC):
    @abstractmethod
    def run(self, args: Dict[str, Any]) -> int:
        pass


# A job class should be the counter part of a job entry on the Quantum circuit database
class QPU(Backend):
    name: str
    pass


class SImulator(Backend):
    name: str
    pass
