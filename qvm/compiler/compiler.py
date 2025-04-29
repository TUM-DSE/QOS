from qiskit.circuit import QuantumCircuit

from qvm.qvm.virtual_circuit import VirtualCircuit
from qvm.qvm.compiler.virtualization import (
    OptimalDecompositionPass,
    GreedyDependencyBreaker,
)
from qvm.qvm.compiler.distr_transpiler import QubitReuser
from qvm.qvm.compiler.types import DistributedTranspilerPass, VirtualizationPass
from qvm.qvm.compiler.util import num_virtual_gates


class QVMCompiler:
    def __init__(
        self,
        virt_passes: list[VirtualizationPass] | None = None,
        dt_passes: list[DistributedTranspilerPass] | None = None,
    ):
        self._virt_passes = virt_passes or []
        self._distributed_transpilers = dt_passes or []

    def run(self, circuit: QuantumCircuit, budget: int) -> VirtualCircuit:
        circuit = circuit.copy()
        for vpass in self._virt_passes:
            if budget == 0:
                break
            elif budget < 0:
                raise ValueError("Compiler failed to keep budget.")
            circuit = vpass.run(circuit, budget)
            budget -= num_virtual_gates(circuit)

        virt_circuit = VirtualCircuit(circuit)
        for dtpass in self._distributed_transpilers:
            dtpass.run(virt_circuit)
        return virt_circuit


class StandardQVMCompiler(QVMCompiler):
    def __init__(self, size_to_reach: int) -> None:
        super().__init__(
            virt_passes=[
                OptimalDecompositionPass(size_to_reach),
                GreedyDependencyBreaker(),
            ],
            dt_passes=[QubitReuser(size_to_reach)],
        )


class CutterCompiler(QVMCompiler):
    def __init__(self, size_to_reach: int) -> None:
        super().__init__([OptimalDecompositionPass(size_to_reach)])
