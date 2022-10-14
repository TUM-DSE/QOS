import itertools
from typing import Dict
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit, Barrier
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
import networkx as nx

from vqc.virtual_gate import VirtualBinaryGate


def dag_to_connectivity_graph(dag: DAGCircuit) -> nx.Graph:
    graph = nx.Graph()
    bb = nx.edge_betweenness_centrality(graph, normalized=False)
    nx.set_edge_attributes(graph, bb, "weight")
    graph.add_nodes_from(dag.qubits)
    for node in dag.op_nodes():
        if isinstance(node.op, VirtualBinaryGate) or isinstance(node.op, Barrier):
            continue
        if len(node.qargs) >= 2:
            for qarg1, qarg2 in itertools.combinations(node.qargs, 2):
                if not graph.has_edge(qarg1, qarg2):
                    graph.add_edge(qarg1, qarg2, weight=0)
                graph[qarg1][qarg2]["weight"] += 1
    return graph


def circuit_to_connectivity_graph(circuit: QuantumCircuit) -> nx.Graph:
    dag = circuit_to_dag(circuit)
    return dag_to_connectivity_graph(dag)


def decompose_virtual_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit = circuit.decompose(
        [
            VirtualBinaryGate,
        ]
    )
    return circuit.decompose(["vgate_end"])


def deflated_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    dag = circuit_to_dag(circuit)
    qubits = set(qubit for qubit in circuit.qubits if qubit not in dag.idle_wires())

    qreg = QuantumRegister(bits=qubits)
    new_circuit = QuantumCircuit(qreg, *circuit.cregs)
    sorted_qubits = sorted(qubits, key=lambda q: circuit.find_bit(q).index)
    qubit_map: Dict[Qubit, Qubit] = {
        q: new_circuit.qubits[i] for i, q in enumerate(sorted_qubits)
    }
    for circ_instr in circuit.data:
        if set(circ_instr.qubits) <= qubits:
            new_circuit.append(
                circ_instr.operation,
                [qubit_map[q] for q in circ_instr.qubits],
                circ_instr.clbits,
            )
    return new_circuit
