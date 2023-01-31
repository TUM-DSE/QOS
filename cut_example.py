# %%
from qvm import cut
from qvm.static import StaticCut
from qvm.static import Qubit

# import qiskit
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from benchmarks.chemistry import HamiltonianSimulationBenchmark

# %%

bench = HamiltonianSimulationBenchmark(4)
circuit = bench.circuit()

circuit.draw()

# %%
qubit3 = circuit._qubits[1]
qubit4 = circuit._qubits[2]
# %%

passes = StaticCut(qubit3, qubit4)

# %%
distcircuit = cut.cut(circuit, passes)
# %%

frags = distcircuit.fragments
print(frags)
frag1 = distcircuit.fragment_as_circuit(frags[0])
frag2 = distcircuit.fragment_as_circuit(frags[1])
# %%
frag1.draw()

# %%

frag2.draw()

# %%
"""
# %%
def get_ideal_counts(circuit: cirq.Circuit) -> Counter:
    ideal_counts = {}
    print(circuit.final_state_vector(ignore_terminal_measurements=True))
    for i, amplitude in enumerate(
        circuit.final_state_vector(ignore_terminal_measurements=True)
    ):
        bitstring = f"{i:>0{len(circuit.all_qubits())}b}"
        probability = np.abs(amplitude) ** 2
        ideal_counts[bitstring] = probability
    return Counter(ideal_counts)


def _get_ideal_counts(circuit: QuantumCircuit) -> Counter:
    ideal_counts = {}
    # sv = Statevector.from_label("0" * circuit.num_qubits)
    # circuit_no_meas = circuit.remove_final_measurements(inplace=False)
    # sv.evolve(circuit_no_meas)
    # sv = StatevectorSimulator(circuit)
    # circuit_no_meas = circuit.remove_final_measurements(inplace=False)
    # sv.evolve(circuit_no_meas)
    # print(sv.run(circuit))
    counts2 = back.result().get_counts(circuit)
    return counts2
"""

# %%
# ham = supermarq.benchmarks.hamiltonian_simulation.HamiltonianSimulation(num_qubits=4)
# marq_circuit = ham.circuit()
# supermark_counts = get_ideal_counts(marq_circuit)
# plot_histogram(supermark_counts)

# %%


# %%
# print(ghz_circuit)
# print(circ)
# plot_gate_map(ghz_circuit)
