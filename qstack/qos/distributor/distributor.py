from typing import Any, Dict, List
from qstack.qernel.qernel import Input, Qernel
from qstack.qos.types import QOSEngineI, QPUWrapper


class Distributor(QOSEngineI):
    """QOS Engine for scheduling qernels on a set of distributed QPUs"""

    _qpus: List[QPUWrapper]

    def __init__(self, qpus: List[QPUWrapper]) -> None:
        self._qpus = qpus

    def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
        pass

    def execute_qernel(self, qid: int, input: Input, exec_args: Dict[str, Any]) -> None:
        pass