from typing import Iterator
import itertools
import jsonpickle as jp
import matplotlib.pyplot as plt

import networkx as nx
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    CircuitInstruction,
    Instruction,
    Qubit,
)


class DAG(nx.DiGraph):
    def __init__(self, circuit: QuantumCircuit = None, dag: nx.DiGraph = None):
        if circuit is not None:
            def _next_op_on_qubit(qubit: int, from_idx: int) -> int:
                for i, instr in enumerate(circuit[from_idx + 1 :]):
                    if qubit in instr.qubits:
                        return i + from_idx + 1
                return -1

            super().__init__()

            for i, instr in enumerate(circuit):
                self.add_node(i, instr=instr)
            
            for qubit in instr.qubits:
                next_op = _next_op_on_qubit(qubit, i)
                if next_op > -1:
                    self.add_edge(i, next_op)

            self._qregs = circuit.qregs
            self._cregs = circuit.cregs

        elif dag is not None:
            super().__init__(dag)
        else:
            super().__init__()

    @classmethod
    def from_circuit(cls, circuit: QuantumCircuit):
        return cls(circuit=circuit)
    
    def draw(self, filename: str = 'dag.png'):
        #nx.draw(self, nx.spring_layout(self. random_state=100), with_labels=True,)
        plt.savefig(filename)
    
    @classmethod
    def from_dag(cls, dag: nx.DiGraph):
        return cls(dag=dag)
    
    @classmethod
    def from_string(cls, dag_str: str):
        return cls.from_dag(cls.deserialize(dag_str))

    @property
    def qubits(self) -> list[Qubit]:
        return list(itertools.chain(*self._qregs))

    @property
    def clbits(self) -> list[Qubit]:
        return list(itertools.chain(*self._cregs))

    @property
    def depth(self) -> int:
        return nx.dag_longest_path_length(self)

    @property
    def to_string(self):
        return jp.encode(nx.json_graph.adjacency_data(self))
    
    @staticmethod
    def deserialize(qc_dag: str):
        return nx.json_graph.adjacency_graph(jp.decode(qc_dag), directed=True)

    def add_qreg(self, qreg: QuantumRegister) -> None:
        if qreg in self._qregs:
            raise ValueError(f"Quantum register {qreg} already exists")
        self._qregs.append(qreg)

    def add_creg(self, creg: ClassicalRegister) -> None:
        if creg in self._cregs:
            raise ValueError(f"Classical register {creg} already exists")
        self._cregs.append(creg)

    def to_circuit(self) -> QuantumCircuit:
        order = list(nx.topological_sort(self))
        circuit = QuantumCircuit(*self._qregs, *self._cregs)
        for i in order:
            instr = self.nodes[i]["instr"]
            circuit.append(instr)
        return circuit

    def add_instr_node(self, instr: CircuitInstruction) -> int:
        new_id = max(self.nodes) + 1 if len(self.nodes) > 0 else 0
        self.add_node(new_id, instr=instr)
        return new_id

    def get_node_instr(self, node: int) -> CircuitInstruction:
        return self.nodes[node]["instr"]

    def qubits_of_edge(self, u: int, v: int) -> set[Qubit]:
        qubits1 = self.get_node_instr(u).qubits
        qubits2 = self.get_node_instr(v).qubits
        return set(qubits1) & set(qubits2)

    def remove_nodes_of_type(self, instr_type: type[Instruction]) -> None:
        nodes_to_remove = []
        for node in self.nodes:
            if isinstance(self.get_node_instr(node).operation, instr_type):
                predecessors = list(self.predecessors(node))
                successors = list(self.successors(node))

                for pred, succ in itertools.product(predecessors, successors):
                    pred_qubits = set(self.get_node_instr(pred).qubits)
                    succ_qubits = set(self.get_node_instr(succ).qubits)
                    if pred_qubits & succ_qubits:
                        self.add_edge(pred, succ)

                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            self.remove_node(node)

    def compact(self) -> None:
        # get the used qubits
        used_qubits: set[Qubit] = set()
        for node in self.nodes:
            used_qubits.update(self.get_node_instr(node).qubits)

        new_qreg = QuantumRegister(len(used_qubits), "q")
        qubit_mapping: dict[Qubit, Qubit] = {
            qubit: new_qreg[i] for i, qubit in enumerate(used_qubits)
        }
        # update the circuit
        for node in self.nodes:
            instr = self.get_node_instr(node)
            new_qubits = [qubit_mapping[qubit] for qubit in instr.qubits]
            instr.qubits = new_qubits

        self._qregs = [new_qreg]

    def instructions_on_qubit(self, qubit: Qubit) -> Iterator[CircuitInstruction]:
        for node in nx.topological_sort(self):
            instr = self.get_node_instr(node)
            if qubit in instr.qubits:
                yield instr

    def nodes_on_qubit(self, qubit: Qubit) -> Iterator[int]:
        for node in nx.topological_sort(self):
            instr = self.get_node_instr(node)
            if qubit in instr.qubits:
                yield node