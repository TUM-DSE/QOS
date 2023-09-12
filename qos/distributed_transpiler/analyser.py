from typing import Any, Dict, List
import random
from qos.types import Engine, Qernel
from qos.distributed_transpiler.types import AnalysisPass
from abc import ABC, abstractmethod
from qos.engines.multiprogrammer import Multiprogrammer
import qos.database as db
from qiskit.providers.fake_provider import *
import mapomatic as mm
import logging
import pdb
import redis
from qiskit import transpile, QuantumCircuit
import numpy as np
import networkx as nx
from qiskit.converters import circuit_to_dag
from qvm.qvm.compiler.dag import *


class BasicAnalysisPass(AnalysisPass):

    def __init__(self) -> None:
        pass

    def name(self) -> str:
        return "BasicAnalysisPass"

    def run(self, qernel: Qernel) -> None:
        qc = qernel.get_circuit()
        simple_metadata = self.get_simple_metadata(qc)
        qernel.edit_metadata(simple_metadata)
    
    def results(self) -> None:
        pass

    def get_simple_metadata(self, qc: QuantumCircuit) -> dict[str, Any]:
        to_return = {}
        to_return["depth"] = qc.depth()
        to_return["num_qubits"] = qc.num_qubits
        to_return["num_nonlocal_gates"] = qc.num_nonlocal_gates()
        to_return['num_connected_components'] = qc.num_connected_components()    
        to_return['number_instructions'] = qc.size()

        for key, value in qc.count_ops().items():
            if key == "measure":
                to_return["num_measurements"] =  value
            elif key == "cx":
                to_return["num_cnot_gates"] = value        

        return to_return
    
class SupermarqFeaturesAnalysisPass(AnalysisPass):

    def __init__(self) -> None:
        pass

    def name(self) -> str:
        return "SupermarqFeaturesAnalysisPass"

    def run(self, qernel: Qernel) -> None:
        qc = qernel.get_circuit()
        supermarq_metadata = self.get_supermarq_metadata(qc)
        qernel.edit_metadata(supermarq_metadata)

    def get_supermarq_metadata(self, qc: QuantumCircuit) -> dict[str, Any]:
        metadata = {}

        metadata["program_communication"] = self.get_programm_communication(qc)
        metadata["liveness"] = self.get_liveness(qc)
        metadata["parallelism"] = self.get_parallelism(qc)
        metadata["measurement"] = self.get_measurement(qc)
        metadata["entanglement_ratio"] = self.get_entanglement_ratio(qc)
        metadata["critical_depth"] = self.get_critical_depth(qc)

        return metadata

    def get_programm_communication(self, q: QuantumCircuit | Qernel) -> float:
        if isinstance(q, Qernel):
            qc = q.circuit()
        else:
            qc = q
        num_qubits = qc.num_qubits
        dag = circuit_to_dag(qc)
        dag.remove_all_ops_named("barrier")

        graph = nx.Graph()
        for op in dag.two_qubit_ops():
            q1, q2 = op.qargs
            graph.add_edge(qc.find_bit(q1).index, qc.find_bit(q2).index)

        degree_sum = sum([graph.degree(n) for n in graph.nodes])

        return degree_sum / (num_qubits * (num_qubits - 1))


    def get_liveness(self, q: QuantumCircuit | Qernel) -> float:
        if isinstance(q, Qernel):
            qc = q.circuit()
        else:
            qc = q

        num_qubits = qc.num_qubits
        dag = circuit_to_dag(qc)
        dag.remove_all_ops_named("barrier")

        activity_matrix = np.zeros((num_qubits, dag.depth()))

        for i, layer in enumerate(dag.layers()):
            for op in layer["partition"]:
                for qubit in op:
                    activity_matrix[qc.find_bit(qubit).index, i] = 1

        return np.sum(activity_matrix) / (num_qubits * dag.depth())


    def get_parallelism(self, q: QuantumCircuit | Qernel) -> float:
        if isinstance(q, Qernel):
            qc = q.circuit()
        else:
            qc = q
        
        dag = circuit_to_dag(qc)
        dag.remove_all_ops_named("barrier")
        return max(1 - (qc.depth() / len(dag.gate_nodes())), 0)


    def get_measurement(self, q: QuantumCircuit | Qernel) -> float:
        if isinstance(q, Qernel):
            qc = q.circuit()
        else:
            qc = q
        
        qc.remove_final_measurements()
        dag = circuit_to_dag(qc)
        dag.remove_all_ops_named("barrier")

        reset_moments = 0
        gate_depth = dag.depth()

        for layer in dag.layers():
            reset_present = False
            for op in layer["graph"].op_nodes():
                if op.name == "reset":
                    reset_present = True
            if reset_present:
                reset_moments += 1

        return reset_moments / gate_depth

    def get_entanglement_ratio(self, q: QuantumCircuit | Qernel) -> float:
        if isinstance(q, Qernel):
            qc = q.circuit()
        else:
            qc = q
        
        dag = circuit_to_dag(qc)
        dag.remove_all_ops_named("barrier")

        return len(dag.two_qubit_ops()) / len(dag.gate_nodes())

    def get_critical_depth(self, q: QuantumCircuit | Qernel) -> float:
        if isinstance(q, Qernel):
            qc = q.circuit()
        else:
            qc = q
        
        dag = circuit_to_dag(qc)
        dag.remove_all_ops_named("barrier")
        n_ed = 0
        two_q_gates = set([op.name for op in dag.two_qubit_ops()])
        for name in two_q_gates:
            try:
                n_ed += dag.count_ops_longest_path()[name]
            except KeyError:
                continue
        n_e = len(dag.two_qubit_ops())

        if n_ed == 0:
            return 0

        return n_ed / n_e

class DAGAnalysisPass(AnalysisPass):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, qernel: Qernel) -> None:
        pass

class DependencyGraphFromDAGPass(DAGAnalysisPass):
    def __init__(self) -> None:
        pass

    def name(self) -> str:
        return "DependencyGraphFromDAGPass"

    def run(self, qernel: Qernel) -> None:
        dag = qernel.get_dag()

        dependency_graph_metadata = {"depenendency_graph" : dag.qubit_dependencies()}

        qernel.edit_metadata(dependency_graph_metadata)

class QubitConnectivityGraphFromDAGPass(DAGAnalysisPass):
    def __init__(self) -> None:
        pass

    def name(self) -> str:
        return "QubitConnectivityGraphFromDAGPass"

    def run(self, qernel: Qernel) -> None:
        dag = qernel.get_dag()

        dependency_graph_metadata = {"qubit_connectivity_graph" : dag_to_qcg(dag, use_qubit_idx=True)}

        qernel.edit_metadata(dependency_graph_metadata)

class QAOAAnalysisPass(DAGAnalysisPass):
    def __init__(self) -> None:
        pass

    def name(self) -> str:
        return "QAOAAnalysisPass"

    def run(self, qernel: Qernel) -> None:
        cgc_pass = QubitConnectivityGraphFromDAGPass()
        cgc_pass.run(qernel)
        cgc = qernel.get_metadata()['qubit_connectivity_graph']

        qaoa_metadata = {
            "h" : self.generate_h_from_graph(cgc),
            "J" : self.generate_J_from_graph(cgc),
            "offset" : 0.0,
            "num_layers" : 1
        }

        qernel.edit_metadata(qaoa_metadata)

    def generate_h_from_graph(self, graph: nx.Graph):
        h = {}

        for node in graph.nodes():
            h[node] = 0.
        
        return h
    
    def generate_J_from_graph(self, graph: nx.Graph):
        J = {}

        for edge in graph.edges:
            weight = random.choice([-1, 1])
            J[edge] = weight

        return J