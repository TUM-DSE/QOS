from abc import ABC, abstractmethod
from typing import Any, Dict

from qstack.qernel import Qernel, Input


class QOSEngineI(ABC):
    """Generic interface for implementations of QOS Layers"""

    @abstractmethod
    def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
        pass

    @abstractmethod
    def execute_qernel(self, qid: int, input: Input, exec_args: Dict[str, Any]) -> None:
        pass


class QPUWrapper(QOSEngineI, ABC):
    @abstractmethod
    def cost(self, qid: int) -> float:
        pass

    @abstractmethod
    def overhead(self, qid: int) -> int:
        pass
