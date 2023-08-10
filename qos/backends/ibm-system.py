import sys
sys.path.insert(0, '/mnt/c/Users/giort/Documents/GitHub/qos/')

from qiskit import IBMQ
from qiskit.providers import *
from qiskit.compiler import transpile
from backends import QPU
from qiskit.circuit import QuantumCircuit

provider = IBMQ.load_account()

backend = QPU('ibm_oslo', provider)

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.measure_all()
#circuit.draw()

qc = transpile(circuit, backend.backend)

qernel = backend.backend.run(qc, shots=8192)

counts = qernel.result().get_counts()
plot_histogram(counts, filename="plot.png")