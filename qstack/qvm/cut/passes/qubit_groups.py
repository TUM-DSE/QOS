import itertools
from typing import Dict, List, Set, Type
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit

from vqc.cut.cut import CutPass, STANDARD_VIRTUAL_GATES
from vqc.virtual_gate.virtual_gate import VirtualBinaryGate


class QubitGroups(CutPass):
    def __init__(
        self,
        groups: List[Set[Qubit]],
        vgates: Dict[str, Type[VirtualBinaryGate]] = STANDARD_VIRTUAL_GATES,
    ):
        assert len(groups) > 0
        self.groups = groups
        super().__init__(vgates)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        print(self.groups)
        for op_node in dag.op_nodes():
            if (
                sum(1 for g in self.groups if set(op_node.qargs) & g) > 1
                and op_node.name != "barrier"
            ):
                dag.substitute_node(
                    op_node,
                    self.vgates[op_node.op.name](op_node.op.params),
                    inplace=True,
                )
        return dag
