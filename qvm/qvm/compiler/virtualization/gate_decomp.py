from networkx.algorithms.community import kernighan_lin_bisection as bisect
from qiskit.circuit import QuantumCircuit, Qubit, Barrier

from qvm.qvm.compiler.util import num_virtual_gates
from qvm.qvm.compiler.asp import qcg_to_asp, get_optimal_symbols
from qvm.qvm.compiler.dag import DAG, dag_to_qcg
from qvm.qvm.compiler.types import VirtualizationPass


class BisectionPass(VirtualizationPass):
    def __init__(self, size_to_reach: int) -> None:
        self._size_to_reach = size_to_reach
        super().__init__()

    def run(self, circuit: QuantumCircuit, budget: int) -> QuantumCircuit:
        dag = DAG(circuit)
        self._recursive_bisection(dag)
        dag.fragment()
        v_circuit = dag.to_circuit()
        if num_virtual_gates(v_circuit) > budget:
            return circuit.copy()
        return dag.to_circuit()

    def _recursive_bisection(self, dag: DAG) -> int:
        qcg = dag_to_qcg(dag)
        partitions: list[set[int]] = [set(dag.qubits)]
        while any(len(f) > self._size_to_reach for f in partitions):
            largest_fragment = max(partitions, key=lambda f: len(f))
            partitions.remove(largest_fragment)
            partitions += list(bisect(qcg.subgraph(largest_fragment)))
        return _decompose_qubit_sets(dag, partitions)
    
    def get_budget(self, circuit: QuantumCircuit) -> int:
        qc_copy = circuit.copy()

        dag = DAG(qc_copy)
        self._recursive_bisection(dag)
        dag.fragment()
        v_circuit = dag.to_circuit()

        return num_virtual_gates(v_circuit)


class OptimalDecompositionPass(VirtualizationPass):
    def __init__(self, size_to_reach: int) -> None:
        self._size_to_reach = size_to_reach
        super().__init__()

    def run(self, circuit: QuantumCircuit, budget: int) -> QuantumCircuit:
        dag = DAG(circuit)
        self._optimal_decomposition(dag)
        dag.fragment()
        v_circuit = dag.to_circuit()
        if num_virtual_gates(v_circuit) > budget:
            return circuit.copy()
        return dag.to_circuit()

    def _asp_code(self, num_partitions: int) -> str:
        asp = f"""
        partition(P) :- P = 0..{num_partitions - 1}.
        
        {{ qubit_in_partition(Qubit, P) : partition(P) }} == 1 :- qubit(Qubit).
        :- partition(P), not qubit_in_partition(_, P).

        virtual_connection(Qubit1, Qubit2, W) :- 
            qubit_in_partition(Qubit1, P1),
            qubit_in_partition(Qubit2, P2),
            Qubit1 != Qubit2,
            P1 != P2,
            qubit_conn(Qubit1, Qubit2, W).

        num_vgates(N) :- N = #sum{{ W, Qubit1, Qubit2 : virtual_connection(Qubit1, Qubit2, W) }}.
        
        num_qubits_in_partition(P, N) :- partition(P), N = #count{{Qubit : qubit_in_partition(Qubit, P)}}.
        :- num_qubits_in_partition(P, N), N > {self._size_to_reach}.

        qubit_diff(P1, P2, D) :- 
            partition(P1), partition(P2), P1 < P2, 
            num_qubits_in_partition(P1, N1), num_qubits_in_partition(P2, N2),
            D = |N1 - N2|.
        
        total_qubit_diff(TD) :- TD = #sum{{ D : qubit_diff(_, _, D) }}.
        
        #minimize{{ 100000@N : num_vgates(N) }}.
        #minimize{{ 1@D : total_qubit_diff(D) }}.
        #show qubit_in_partition/2.
        """
        return asp

    def _optimal_decomposition(self, dag: DAG) -> int:
        qcg = dag_to_qcg(dag, use_qubit_idx=True)
        asp = qcg_to_asp(qcg)
        num_partitions = qcg.number_of_nodes() // self._size_to_reach + (
            qcg.number_of_nodes() % self._size_to_reach != 0
        )
        num_partitions = max(2, num_partitions)
        asp += self._asp_code(num_partitions=num_partitions)
        symbols = get_optimal_symbols(asp)
        qubit_sets: list[set[Qubit]] = [set() for _ in range(num_partitions)]
        for symbol in symbols:
            if symbol.name != "qubit_in_partition":
                continue
            qubit_idx = symbol.arguments[0].number
            partition_idx = symbol.arguments[1].number
            qubit_sets[partition_idx].add(dag.qubits[qubit_idx])
        return _decompose_qubit_sets(dag, qubit_sets)

    def get_budget(self, circuit: QuantumCircuit) -> int:
        qc_copy = circuit.copy()

        dag = DAG(qc_copy)
        self._optimal_decomposition(dag)
        dag.fragment()
        v_circuit = dag.to_circuit()

        return num_virtual_gates(v_circuit)

def _decompose_qubit_sets(dag: DAG, qubit_sets: list[set[Qubit]]) -> int:
    vgates = 0
    for node in dag.nodes:
        instr = dag.get_node_instr(node)
        qubits = instr.qubits

        nums_of_frags = sum(1 for qubit_set in qubit_sets if set(qubits) & qubit_set)
        if nums_of_frags == 0:
            raise ValueError(f"No fragment found for qubit {qubits}.")
        elif nums_of_frags > 1 and not isinstance(instr.operation, Barrier):
            dag.virtualize_node(node)
            vgates += 1
    return vgates
