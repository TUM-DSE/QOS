from typing import Any, Dict
from qstack.qernel.qernel import Input, Qernel
from qstack.qos.types import QOSEngineI


class Merger(QOSEngineI):
    """QOS Engine for merging qernels to one qernel if possible to save execution time"""

    def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
        pass

    def execute_qernel(self, qid: int, input: Input, exec_args: Dict[str, Any]) -> None:
        pass
