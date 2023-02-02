from typing import Dict, Type
from qiskit.dagcircuit import DAGCircuit

from vqc.cut.cut import CutPass, STANDARD_VIRTUAL_GATES
from vqc.virtual_gate.virtual_gate import VirtualBinaryGate
from .qubit_groups import QubitGroups


class LadderDecomposition(CutPass):
    def __init__(
        self,
        num_partitions: int,
        vgates: Dict[str, Type[VirtualBinaryGate]] = STANDARD_VIRTUAL_GATES,
    ) -> None:
        self.num_partitions = num_partitions
        super().__init__(vgates)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        num_frags = min(self.num_partitions, dag.num_qubits())
        frag_size = dag.num_qubits() // num_frags
        groups = []
        for i in range(num_frags):
            groups.append(set(dag.qubits[i * frag_size : (i + 1) * frag_size]))
        return QubitGroups(groups, self.vgates).run(dag)
