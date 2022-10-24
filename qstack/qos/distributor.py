from typing import Any, Dict, List

from qstack.qernel.qernel import Qernel, QernelArgs
from qstack.types import QOSEngineI, QPUWrapper


class Distributor(QOSEngineI):
    """QOS Engine for scheduling qernels on a set of distributed QPUs"""

    _qpus: List[QPUWrapper]

    def __init__(self, qpus: List[QPUWrapper]) -> None:
        self._qpus = qpus

    def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
        pass

    def execute_qernel(
        self, qid: int, input: QernelArgs, exec_args: Dict[str, Any]
    ) -> None:
        pass
