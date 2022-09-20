from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.dagcircuit import DAGCircuit

from qos.qvm.cut.cut import CutPass
from qos.qvm.converters import dag_to_connectivity_graph
from .qubit_groups import QubitGroups


class Bisection(CutPass):
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        cg = dag_to_connectivity_graph(dag)
        A, B = kernighan_lin_bisection(cg)
        return QubitGroups([A, B], self.vgates).run(dag)
