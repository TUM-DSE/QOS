from qos.types import Qernel
from qos.error_mitigator.analyser import BasicAnalysisPass
from qiskit import QuantumCircuit

from qiskit_ibm_runtime import IBMBackend
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPSchedule
from typing import Dict, List, Any

import networkx as nx

def get_all_backend_info(backend: IBMBackend) -> Dict[str, Any]:
    """
    Retrieve all relevant information about a backend using the IBMBackend API.

    Args:
        backend (IBMBackend): The IBM Qiskit backend instance.

    Returns:
        Dict[str, Any]: A dictionary containing backend information, including:
            - Coupling map
            - Backend properties
            - Number of qubits
            - Basis gates
            - Quantum volume
            - Maximum circuits
            - Supported features
    """
    backend_info = {
        "name": backend.name,
        "num_qubits": backend.num_qubits,
        "coupling_map": backend.coupling_map,
        "basis_gates": backend.configuration().basis_gates,
        "quantum_volume": backend.configuration().quantum_volume,
        "max_circuits": backend.configuration().max_experiments,
        "simulator": backend.configuration().simulator,
        "backend_version": backend.configuration().backend_version,
        "properties": backend.properties().to_dict() if backend.properties() else None,
        "supported_features": backend.configuration().supported_instructions,
    }

    return backend_info

def compute_qubit_utility(backend_properties: Dict[str, Any]) -> Dict[int, float]:
    """
    Compute the utility of each physical qubit on a QPU.

    Args:
        coupling_map (List[List[int]]): The coupling map of the QPU, where each entry is a pair of connected qubits.
        backend_properties (BackendProperties): The backend properties containing error rates for the QPU.

    Returns:
        Dict[int, float]: A dictionary mapping each qubit to its computed utility value.
    """
    # Initialize utility dictionary
    utility = {}
    coupling_map = backend_properties['coupling_map']

    # Iterate over each qubit in the coupling map
    for qubit in range(len(backend_properties.qubits)):
        # Get the neighbors of the qubit from the coupling map
        neighbors = [pair[1] for pair in coupling_map if pair[0] == qubit] + \
                    [pair[0] for pair in coupling_map if pair[1] == qubit]

        # Compute the number of links (degree of the qubit)
        num_links = len(neighbors)

        # Compute the sum of error rates for the links
        error_sum = 0
        for neighbor in neighbors:
            try:
                # Get the error rate for the link (CX gate error)
                error_rate = backend_properties.gate_error('cx', [qubit, neighbor])
                error_sum += error_rate
            except:
                # If no error rate is available, assume a default value
                error_sum += 0.01  # Default error rate

        # Compute the utility for the qubit
        if error_sum > 0:
            utility[qubit] = num_links / error_sum
        else:
            utility[qubit] = 0  # Avoid division by zero

    return utility

def compute_CMR(circuit: QuantumCircuit) -> float:
    """
    Analyze the circuit using BasicAnalysisPass and compute the ratio of nonlocal gates to measurements.

    Args:
        circuit (QuantumCircuit): The quantum circuit.

    Returns:
        float: The ratio of nonlocal gates to measurements.
    """
    # Create a Qernel from the circuit
    qernel = Qernel(circuit)

    # Run BasicAnalysisPass
    analysis_pass = BasicAnalysisPass()
    analysis_pass.run(qernel)

    # Extract metadata
    metadata = qernel.get_metadata()
    num_nonlocal_gates = metadata.get("num_nonlocal_gates", 0)
    num_measurements = metadata.get("num_measurements", 1)  # Avoid division by zero

    return num_nonlocal_gates / num_measurements

def analyze_programs(programs: List[QuantumCircuit]) -> Dict[int, Dict[str, Any]]: 
    """
    Analyze a list of quantum programs to compute usage, interaction, and CMR for each program qubit.

    Args:
        programs (List[QuantumCircuit]): A list of quantum circuits (programs).

    Returns:
        Dict[int, Dict[str, Any]]: A dictionary where each program index maps to its analysis results, including:
            - Usage: A dictionary mapping each qubit to its usage ratio.
            - Interaction: A dictionary mapping each qubit to the set of qubits it interacts with.
            - CMR: The compute-to-measurement ratio for the program.
            - Ranked Qubits: A list of qubits ranked by their usage.
    """
    program_analysis = {}

    for program_index, circuit in enumerate(programs):
        # Initialize usage and interaction dictionaries
        usage = {}
        interaction = {}

        # Analyze the circuit
        total_instructions = circuit.size()
        for qubit in circuit.qubits:
            # Compute usage for each qubit
            instructions_using_qubit = sum(1 for instr in circuit.data if qubit in instr.qubits)
            usage[qubit] = instructions_using_qubit / total_instructions

            # Compute interaction for each qubit
            interacting_qubits = set()
            for instr in circuit.data:
                if qubit in instr.qubits:
                    interacting_qubits.update(instr.qubits)
            interacting_qubits.discard(qubit)  # Remove the qubit itself
            interaction[qubit] = interacting_qubits

        # Compute CMR for the program
        cmr = compute_CMR(circuit)

        # Rank qubits by usage
        ranked_qubits = sorted(usage.keys(), key=lambda q: usage[q], reverse=True)

        # Store the analysis results for the program
        program_analysis[program_index] = {
            "Usage": usage,
            "Interaction": interaction,
            "CMR": cmr,
            "Ranked Qubits": ranked_qubits,
        }

    return program_analysis

def create_sub_graph(circuit: QuantumCircuit, utility: dict, cmr: float, alpha: float, beta: float) -> nx.Graph:
    """
    Locate a reliable cluster on the chip and create a subgraph.

    Args:
        circuit (QuantumCircuit): The quantum circuit (program).
        utility (dict): A dictionary mapping qubits to their utility values.
        cmr (float): Circuit Measurement Reliability threshold.
        alpha (float): Percentage of neighbors with high utility.
        beta (float): Percentage of nodes with measurement errors below the mean.

    Returns:
        nx.Graph: The subgraph representing the reliable cluster.
    """
    # Initialize graph and rank
    graph = nx.Graph()
    for qubit, neighbors in utility.items():
        for neighbor in neighbors:
            graph.add_edge(qubit, neighbor)

    rank = 0
    root_node = None

    # Find the root node
    while root_node is None:
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            high_utility_neighbors = [n for n in neighbors if utility[n] > alpha]
            low_error_nodes = [n for n in neighbors + [node] if utility[n] < beta]

            if len(high_utility_neighbors) >= alpha * len(neighbors) and len(low_error_nodes) >= beta * len(neighbors + [node]):
                root_node = node
                break

        if root_node is None:
            rank += 1

    # Grow the subgraph
    sub_graph = nx.Graph()
    sub_graph.add_node(root_node)
    boundary = list(graph.neighbors(root_node))

    while len(sub_graph.nodes) < circuit.num_qubits:
        for node in boundary:
            if len(sub_graph.nodes) >= circuit.num_qubits:
                break
            if utility[node] < beta:
                continue
            sub_graph.add_node(node)
            sub_graph.add_edges_from([(node, neighbor) for neighbor in graph.neighbors(node) if neighbor in sub_graph.nodes])
            boundary.remove(node)
            boundary.extend([n for n in graph.neighbors(node) if n not in sub_graph.nodes and n not in boundary])

    return sub_graph

def fair_and_reliable_partition(graph: nx.Graph, usage: dict, interaction: dict, circuit: QuantumCircuit) -> dict:
    """
    Perform qubit allocation for fair and reliable partitioning.

    Args:
        graph (nx.Graph): The subgraph representing the reliable cluster.
        usage (dict): A dictionary mapping program qubits to their usage.
        interaction (dict): A dictionary mapping program qubits to their interactions.
        circuit (QuantumCircuit): The quantum circuit (program).

    Returns:
        dict: The qubit allocation.
    """
    allocation = {}
    unmapped_ancilla = [q for q in circuit.qubits if q not in usage]
    unmapped_program = [q for q in circuit.qubits if q in usage]

    # Map ancilla qubits
    for ancilla in unmapped_ancilla:
        for phy_q in graph.nodes:
            if phy_q not in allocation.values():
                allocation[ancilla] = phy_q
                break

    # Map program qubits
    for program_qubit in unmapped_program:
        for phy_q in sorted(graph.nodes, key=lambda x: usage.get(x, 0), reverse=True):
            if phy_q not in allocation.values():
                allocation[program_qubit] = phy_q
                break

    return allocation

def independent_qubit_allocation_and_scheduling(
    programs: List[QuantumCircuit],
    program_analysis: Dict[int, Dict[str, Any]],
    backend_props: Dict[int, float]
) -> List[QuantumCircuit]:
    """
    Perform independent qubit allocation and scheduling for a list of quantum programs.

    Args:
        programs (List[QuantumCircuit]): A list of quantum circuits (programs).
        program_analysis (Dict[int, Dict[str, Any]]): Analysis results for each program, including:
            - Usage: A dictionary mapping each qubit to its usage ratio.
            - Interaction: A dictionary mapping each qubit to the set of qubits it interacts with.
            - CMR: The compute-to-measurement ratio for the program.
            - Ranked Qubits: A list of qubits ranked by their usage.
        coupling_map (List[List[int]]): The coupling map of the QPU.

    Returns:
        List[QuantumCircuit]: A list of scheduled quantum circuits after applying SABRE mapping.
    """
    scheduled_programs = []

    for program_index, program in enumerate(programs):
        # Extract analysis results for the current program
        analysis = program_analysis[program_index]
        usage = analysis["Usage"]
        interaction = analysis["Interaction"]
        cmr = analysis["CMR"]

        # Step 1: Create a subgraph for the program
        graph = create_sub_graph(program, usage, cmr, alpha=0.6, beta=0.4)  # Alpha and beta are fixed here

        # Step 2: Perform fair and reliable partitioning
        qubit_allocation = fair_and_reliable_partition(graph, usage, interaction, program)

        # Step 3: Apply SABRE mapping for variation-aware scheduling
        coupling = backend_props['coupling_map']
        sabre_swap = SabreSwap(coupling)
        pass_manager = PassManager(sabre_swap)
        scheduled_program = pass_manager.run(program)

        # Append the scheduled program to the result list
        scheduled_programs.append(scheduled_program)

    return scheduled_programs

def shared_qubit_allocation_and_scheduling(
    programs: List[QuantumCircuit],
    program_analysis: Dict[int, Dict[str, Any]],
    backend_props: Dict[str, Any]
) -> List[QuantumCircuit]:
    """
    Perform shared qubit allocation and scheduling for a list of quantum programs.

    Args:
        programs (List[QuantumCircuit]): A list of quantum circuits (programs).
        program_analysis (Dict[int, Dict[str, Any]]): Analysis results for each program, including:
            - Usage: A dictionary mapping each qubit to its usage ratio.
            - Interaction: A dictionary mapping each qubit to the set of qubits it interacts with.
            - CMR: The compute-to-measurement ratio for the program.
            - Ranked Qubits: A list of qubits ranked by their usage.
        backend_props (Dict[str, Any]): Backend properties, including the coupling map.

    Returns:
        List[QuantumCircuit]: A list of scheduled quantum circuits after applying SABRE mapping.
    """
    scheduled_programs = []
    shared_allocation = {}  # Shared allocation across all programs

    # Step 1: Create a combined subgraph for all programs
    combined_graph = nx.Graph()
    for program_index, program in enumerate(programs):
        analysis = program_analysis[program_index]
        usage = analysis["Usage"]
        cmr = analysis["CMR"]

        # Create a subgraph for the current program
        program_graph = create_sub_graph(program, usage, cmr, alpha=0.6, beta=0.4)
        combined_graph = nx.compose(combined_graph, program_graph)

    # Step 2: Perform shared qubit allocation
    for program_index, program in enumerate(programs):
        analysis = program_analysis[program_index]
        usage = analysis["Usage"]
        interaction = analysis["Interaction"]

        # Allocate qubits for the current program
        allocation = fair_and_reliable_partition(combined_graph, usage, interaction, program)

        # Merge the allocation into the shared allocation
        shared_allocation.update(allocation)

    # Step 3: Apply SABRE mapping for variation-aware scheduling
    coupling_map = backend_props['coupling_map']
    sabre_swap = SabreSwap(coupling_map)
    pass_manager = PassManager(sabre_swap)

    for program in programs:
        # Apply SABRE mapping to each program
        scheduled_program = pass_manager.run(program)
        scheduled_programs.append(scheduled_program)

    return scheduled_programs

def generate_schedule(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Generate a schedule for the given quantum circuit using ALAP scheduling.

    Args:
        circuit (QuantumCircuit): The quantum circuit to schedule.

    Returns:
        QuantumCircuit: The scheduled quantum circuit.
    """
    pass_manager = PassManager(ALAPSchedule())
    scheduled_circuit = pass_manager.run(circuit)
    return scheduled_circuit

def shared_scheduling_with_error_check(
    programs: List[QuantumCircuit],
    backend_props: Dict[str, Any],
    tolerance: float
) -> None:
    """
    Perform shared scheduling for a list of quantum programs and check for mean error rate warnings.

    Args:
        programs (List[QuantumCircuit]): A list of quantum circuits (programs).
        backend_props (Dict[str, Any]): Backend properties, including error rates and coupling map.
        tolerance (float): The error tolerance threshold for generating warnings.

    Returns:
        None
    """
    # Step 1: Generate schedules for all programs
    global_schedule = []
    for program in programs:
        scheduled_program = generate_schedule(program)
        global_schedule.append(scheduled_program)

    # Step 2: Check mean error rate for each program
    for program_index, program in enumerate(programs):
        coupling_map = backend_props["coupling_map"]
        error_rates = backend_props["properties"]["gate_errors"]

        # Calculate the mean error rate for all links in the program's coupling graph
        mean_error_rate = 0
        num_links = 0
        for edge in coupling_map:
            qubit1, qubit2 = edge
            if (qubit1, qubit2) in error_rates:
                mean_error_rate += error_rates[(qubit1, qubit2)]
                num_links += 1

        if num_links > 0:
            mean_error_rate /= num_links

        # Check if the mean error rate exceeds the tolerance threshold
        if mean_error_rate * tolerance < backend_props["shared_error_threshold"]:
            print(f"Warning: Program {program_index} has a mean error rate below the shared threshold.")

    return global_schedule