import networkx as nx

from clingo.solving import Symbol
from clingo.control import Control
from qiskit.circuit import Barrier

from .dag import DAG


def dag_to_asp(dag: DAG) -> str:
    dag.remove_nodes_of_type(Barrier)
    qubits = dag.qubits
    asp = ""
    for node in dag.nodes:
        instr = dag.get_node_instr(node)
        asp += f"gate({node}).\n"
        qubits = instr.qubits
        for qubit in dag.get_node_instr(node).qubits:
            asp += f"gate_on_qubit({node}, {dag.qubits.index(qubit)}).\n"

        for next_node in dag.successors(node):
            next_qubits = dag.get_node_instr(next_node).qubits
            same_qubits = set(qubits) & set(next_qubits)
            if len(same_qubits) == 0:
                raise Exception("No common qubits")
            for qubit in same_qubits:
                asp += f"wire({dag.qubits.index(qubit)}, {node}, {next_node}).\n"
    return asp


def qcg_to_asp(qcg: nx.Graph) -> str:
    asp = ""
    for node, data in qcg.nodes(data=True):
        asp += f"qubit({node}).\n"
    for u, v, data in qcg.edges(data=True):
        if "weight" not in data:
            asp += f"qubit_conn({u}, {v}, 1).\n"
        else:
            asp += f'qubit_conn({u}, {v}, {data["weight"]}).\n'
    return asp


def get_optimal_symbols(asp: str) -> list[Symbol]:
    control = Control()
    control.configuration.solve.models = 0  # type: ignore
    control.add("base", [], asp)
    control.ground([("base", [])])
    solve_result = control.solve(yield_=True)  # type: ignore
    opt_model = None
    for model in solve_result:  # type: ignore
        opt_model = model

    if opt_model is None:
        raise ValueError("No solution found.")

    return list(opt_model.symbols(shown=True))  # type: ignore
