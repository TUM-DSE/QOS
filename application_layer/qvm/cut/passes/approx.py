from typing import Dict, Type

from qiskit.dagcircuit import DAGCircuit

from vqc.cut.cut import CutPass
from vqc.virtual_gate import (
    VirtualBinaryGate,
    VirtualCX,
    VirtualCZ,
    VirtualRZZ,
    ApproxVirtualCX,
    ApproxVirtualCZ,
    ApproxVirtualRZZ,
)

STANDARD_APPROXIMATIONS = {
    VirtualCZ: ApproxVirtualCZ,
    VirtualCX: ApproxVirtualCX,
    VirtualRZZ: ApproxVirtualRZZ,
}


class Approximation(CutPass):
    def __init__(
        self,
        approx_gates: Dict[
            Type[VirtualBinaryGate], Type[VirtualBinaryGate]
        ] = STANDARD_APPROXIMATIONS,
    ):
        self.approx_gates = approx_gates
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.op_nodes():
            if type(node.op) in self.approx_gates:
                approx_gate = self.approx_gates[type(node.op)](*node.op.params)
                dag.substitute_node(node, approx_gate, inplace=True)
