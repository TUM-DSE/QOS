import sys
sys.path.insert(0, '/mnt/c/Users/giort/Documents/GitHub/qos/')

from qiskit import IBMQ
from qiskit.providers import *
from backends import QPU

provider = IBMQ.load_account()
#IBMQ.get_provider(project='main')

backend = QPU('ibm_nairobi', provider)

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(0, 2)
circuit.measure_all()
#circuit.draw()


job = execute(qc, backend, shots=8192)

counts = job.result().get_counts()
plot_histogram(counts, filename="plot.png")