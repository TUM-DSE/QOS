from typing import Any, Dict
from qiskit_aer import AerSimulator

from qstack.qernel import Qernel, QernelArgs
from qstack.types import QPUWrapper


class IBMQQPU(QPUWrapper):
    _qernels: Dict[int, Qernel]
    _qid_ctr: int
    _backend: AerSimulator

    def __init__(self) -> None:
        self._backend = AerSimulator()
        self._qernels = {}
        self._qid_ctr = 0

    def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
        self._qernels[self._qid_ctr] = qernel
        self._qid_ctr += 1
        return self._qid_ctr - 1

    def execute_qernel(self, qid: int, args: QernelArgs, shots: int) -> None:
        qernel = self._qernels[qid]
        circ = qernel.with_input(args=args)

    def cost(self, qid: int) -> float:
        pass

    def overhead(self, qid: int) -> int:
        pass
