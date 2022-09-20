from typing import Any, Dict, List, Optional

from qiskit import QuantumCircuit
from qiskit.providers.ibmq import AccountProvider

from qos.qvm.cut import CutPass, Bisection, cut
from qos.qvm.device import IBMQDevice
from qos.qvm.executor import execute


class LargeIBMQSimulator:
    def __init__(
        self,
        provider: AccountProvider,
        cut_passes: Optional[List[CutPass]],
        transpiler_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.backend = provider.get_backend("ibmq_qasm_simulator")
        if cut_passes is None:
            cut_passes = [Bisection()]
        self.cut_passes = cut_passes
        if transpiler_options is None:
            transpiler_options = {}
        self.transpiler_options = transpiler_options

    def run(self, circuit: QuantumCircuit, shots: int = 10000) -> Dict[str, int]:
        cutted_circ = cut(circuit, *self.cut_passes)
        frags = cutted_circ.fragments
        for frag in frags:
            cutted_circ.set_fragment_device(
                frag, IBMQDevice(self.backend, self.transpiler_options)
            )
        return execute(cutted_circ, shots=shots)
