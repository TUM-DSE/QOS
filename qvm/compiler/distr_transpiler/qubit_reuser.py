from typing import Iterator
from itertools import permutations

import networkx as nx
from qiskit.circuit import Qubit, CircuitInstruction, Reset, Measure
from qiskit.circuit.library.standard_gates import XGate

from qvm.compiler.dag import DAG
from qvm.compiler.types import DistributedTranspilerPass
from qvm.virtual_circuit import VirtualCircuit


class QubitReuser(DistributedTranspilerPass):
    def __init__(self, size_to_reach: int, dynamic: int = True) -> None:
        self._size_to_reach = size_to_reach
        self._dynamic = dynamic
        super().__init__()

    def run(self, virt: VirtualCircuit) -> None:
        frag_circs = virt.fragment_circuits.items()
        for frag, frag_circ in frag_circs:
            dag = DAG(frag_circ)
            random_qubit_reuse(dag, self._size_to_reach)
            if self._dynamic:
                dynamic_measure_and_reset(dag)
            virt.replace_fragment_circuit(frag, dag.to_circuit())


def dynamic_measure_and_reset(dag: DAG) -> None:
    """Converts measure-resets to a measure and a dynamic conditional X gate.

    Args:
        dag (DAG): The DAG to modify.
    """
    nodes = list(dag.nodes())
    for node in nodes:
        instr = dag.get_node_instr(node)

        if not isinstance(instr.operation, Measure):
            continue

        clbit = instr.clbits[0]

        next_node = next(dag.successors(node), None)
        if next_node is None:
            continue

        next_instr = dag.get_node_instr(next_node)
        if not isinstance(next_instr.operation, Reset):
            continue

        next_instr.operation = XGate().c_if(clbit, 1)


def random_qubit_reuse(dag: DAG, size_to_reach: int = 1) -> None:
    num_qubits = len(dag.qubits)
    while num_qubits > size_to_reach:
        print("entered")
        qubit_pair = next(find_valid_reuse_pairs(dag), None)
        if qubit_pair is None:
            break
        reuse(dag, *qubit_pair)
        dag.compact()
        num_qubits -= 1


def reuse(dag: DAG, qubit: Qubit, reused_qubit: Qubit) -> None:
    """
    Reuse a qubit by resetting it and reusing it.
    NOTE: Only works if qubit is not dependent on reused_qubit.
        This must be checked by the caller.

    Args:
        dag (DAG): The DAG to modify.
        qubit (Qubit): The qubit.
        reused_qubit (Qubit): The qubit to reuse.
    """

    # first op of u_qubit
    first_node = next(dag.nodes_on_qubit(reused_qubit))
    # last op of v_qubit
    last_node = list(dag.nodes_on_qubit(qubit))[-1]

    reset_instr = CircuitInstruction(operation=Reset(), qubits=(reused_qubit,))
    reset_node = dag.add_instr_node(reset_instr)
    dag.add_edge(last_node, reset_node)
    dag.add_edge(reset_node, first_node)

    for node in dag.nodes:
        instr = dag.get_node_instr(node)
        instr.qubits = [
            reused_qubit if instr_qubit == qubit else instr_qubit
            for instr_qubit in instr.qubits
        ]


def is_dependent_qubit(dag: DAG, u_qubit: Qubit, v_qubit: Qubit) -> bool:
    """Checks if any operation on u_qubit depends on any operation on v_qubit.

    Args:
        dag (DAG): The DAG to check.
        u_qubit (Qubit): The first qubit.
        v_qubit (Qubit): The second qubit.

    Returns:
        bool: Whether any operation on u_qubit depends on any operation on v_qubit.
    """
    # first op of u_qubit
    u_node = next(dag.nodes_on_qubit(u_qubit))
    # last op of v_qubit
    v_node = list(dag.nodes_on_qubit(v_qubit))[-1]
    return nx.has_path(dag, u_node, v_node)


def find_valid_reuse_pairs(dag: DAG) -> Iterator[tuple[Qubit, Qubit]]:
    """Finds all valid reuse pairs in a DAG by trying every possible pair. O(n^2).

    Args:
        dag (DAG): The DAG to check.

    Yields:
        Iterator[tuple[Qubit, Qubit]]: All valid reuse pairs.
    """
    for qubit, reused_qubit in permutations(dag.qubits, 2):
        if not is_dependent_qubit(dag, reused_qubit, qubit):
            yield qubit, reused_qubit
