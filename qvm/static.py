from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit

from vqc.cut.cut import cut_qubit_connection, CutPass


class StaticCut(CutPass):
    def __init__(self, qubit1: Qubit, qubit2: Qubit):
        self.qubit1 = qubit1
        self.qubit2 = qubit2
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        cut_qubit_connection(dag, self.qubit1, self.qubit2)
        return dag
