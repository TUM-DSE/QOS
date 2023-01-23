from qiskit import IBMQ
from qiskit.providers import fake_provider
from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit.tools.visualization import plot_histogram

backend = fake_provider.FakeManilaV2()

# Create a simple circuit
circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(0, 2)
circuit.measure_all()
circuit.draw()

# Transpile the ideal circuit to a circuit that can be directly executed by the backend
transpiled_circuit = transpile(circuit, backend)
transpiled_circuit.draw()

# Run the transpiled circuit using the simulated fake backend
job = backend.run(transpiled_circuit)
counts = job.result().get_counts()
plot_histogram(counts, filename="plot.png")
