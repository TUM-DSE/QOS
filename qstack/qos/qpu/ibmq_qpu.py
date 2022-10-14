from typing import Any, Dict
from qiskit.providers.ibmq import AccountProvider

from qstack.qernel.qernel import Input, Qernel
from qstack.qos.types import QPUWrapper


class IBMQQPU(QPUWrapper):
    def __init__(self, provider: AccountProvider, backend_name: str) -> None:
        self._backend = provider.get_backend(backend_name)

    def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
        pass

    def execute_qernel(self, qid: int, input: Input, exec_args: Dict[str, Any]) -> None:
        pass

    def cost(self, qid: int) -> float:
        pass

    def overhead(self, qid: int) -> int:
        pass
