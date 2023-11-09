import networkx as nx
from qiskit.circuit import QuantumCircuit, QuantumRegister, CircuitInstruction, Qubit
from qiskit.circuit.library import SwapGate

from qvm.qvm.compiler.types import VirtualizationPass
from qvm.qvm.compiler.dag import DAG
from qvm.qvm.compiler.asp import dag_to_asp, get_optimal_symbols
from qvm.qvm.virtual_gates import WireCut, VirtualMove
from qvm.qvm.compiler.util import num_virtual_gates


class OptimalWireCutter(VirtualizationPass):
    def __init__(self, size_to_reach: int) -> None:
        self._size_to_reach = size_to_reach
        super().__init__()

    def run(self, circuit: QuantumCircuit, budget: int) -> QuantumCircuit:
        dag = DAG(circuit)
        num_cuts = self._cut_wires(dag)
        new_circuit = dag.to_circuit()
        self._wire_cuts_to_moves(dag, num_cuts)
        dag.fragment()
        new_circuit = dag.to_circuit()
        if num_virtual_gates(new_circuit) > budget:
            raise ValueError("Could not find a solution (internal error)")
        return new_circuit

    def _cut_wires(self, dag: DAG) -> int:
        min_num_fragments = len(dag.qubits) // self._size_to_reach
        min_num_fragments = max(min_num_fragments, 2)
        partitions: dict[int, int] | None = None
        edges = list(dag.edges())
        while partitions is None:
            if min_num_fragments > len(dag.qubits):
                raise ValueError("Could not find a solution (internal error)")
            partitions = self._find_optimal_partitons(dag, min_num_fragments)
            min_num_fragments += 1
        #print("found partitions")
        edges = list(dag.edges())
        i = 0
        vgates = 0
        for u, v in edges:
            if partitions[u] != partitions[v]:
                i += 1
                dag.remove_edge(u, v)
                qubits = set(dag.get_node_instr(u).qubits) & set(
                    dag.get_node_instr(v).qubits
                )                
                for qubit in qubits:
                    new_instr = CircuitInstruction(WireCut(), [qubit])
                    w = dag.add_instr_node(new_instr)
                    dag.add_edge(u, w)
                    dag.add_edge(w, v)
                    vgates += 1
        return vgates

    def _wire_cuts_to_moves(self, dag: DAG, num_wire_cuts: int) -> None:
        move_reg = QuantumRegister(num_wire_cuts, "vmove")
        dag.add_qreg(move_reg)

        qubit_mapping: dict[Qubit, Qubit] = {}

        def _find_qubit(qubit: Qubit) -> Qubit:
            while qubit in qubit_mapping:
                qubit = qubit_mapping[qubit]
            return qubit

        cut_ctr = 0
        for node in nx.topological_sort(dag):
            instr = dag.get_node_instr(node)
            instr.qubits = [_find_qubit(qubit) for qubit in instr.qubits]
            if isinstance(instr.operation, WireCut):
                instr.operation = VirtualMove(SwapGate())
                instr.qubits.append(move_reg[cut_ctr])
                qubit_mapping[instr.qubits[0]] = instr.qubits[1]
                cut_ctr += 1

    def _find_optimal_partitons(
        self, dag: DAG, num_fragments: int
    ) -> dict[int, int] | None:
        asp = dag_to_asp(dag)
        asp += self._asp_code(num_fragments=num_fragments)

        try:
            symbols = get_optimal_symbols(asp)
        except ValueError:
            return None

        partitions: dict[int, int] = {}
        for symbol in symbols:
            if symbol.name != "gate_in_partition":
                continue
            gate_idx = symbol.arguments[0].number
            partition_idx = symbol.arguments[1].number
            partitions[gate_idx] = partition_idx
        return partitions

    def _asp_code(self, num_fragments: int) -> str:
        asp = f"""
        partition(P) :- P = 0..{num_fragments - 1}.

        {{ gate_in_partition(Gate, P) : partition(P) }} == 1 :- gate(Gate).
        :- partition(P), not gate_in_partition(_, P).
        
        cutted_wire(Qubit, Gate1, Gate2) :- 
            wire(Qubit, Gate1, Gate2), 
            gate_in_partition(Gate1, P1), 
            gate_in_partition(Gate2, P2), 
            P1 != P2.
            
        num_cutted_wires(N) :- N = #count{{Qubit, Gate1, Gate2 : cutted_wire(Qubit, Gate1, Gate2)}}.
        
        qubit_in_partition(Q, P) :- gate_in_partition(Gate, P), gate_on_qubit(Gate, Q).
        num_qubits_in_partition(P, N) :- partition(P), N = #count{{Q : qubit_in_partition(Q, P)}}.
        :- num_qubits_in_partition(P, N), N > {self._size_to_reach}.
        
        #minimize{{ N : num_cutted_wires(N) }}.
        
        #show gate_in_partition/2.
        #show num_cutted_wires/1.
        """
        return asp

    def get_budget(self, circuit: QuantumCircuit) -> int:
        qc_copy = circuit.copy()

        dag = DAG(qc_copy)
        num_cuts = self._cut_wires(dag)
        self._wire_cuts_to_moves(dag, num_cuts)
        dag.fragment()
        new_circuit = dag.to_circuit()

        return num_virtual_gates(new_circuit)