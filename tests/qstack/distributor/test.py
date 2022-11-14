import pdb
import sys
import logging

sys.path.insert(0, "./../../../.")
from qiskit import QuantumCircuit, QuantumRegister

from qstack.qernel.qernel import Qernel
from qstack.backends.test_qpu import TestQPU
from qstack.backends._ibm import IBMQQPU
from qstack.qos.distributor import Distributor
from qiskit.circuit.random.utils import random_circuit
from qiskit import IBMQ

# from qstack.types import

def generate_jobs(dist_engine, n_jobs: int):
    circuit = random_circuit(num_qubits=3, depth=2)
    #qernel = Qernel(circuit)
    #jobs = n_jobs * [qernel]
    jobs = n_jobs * [circuit]
    return jobs
 
"""
def list_jobs(dist_engine):
	print("There are currently:", dist_engine.job_counter, "jobs on the queue")

	#for i in range(dist_engine.job_counter):
	#   print("Job", i, ":", dist_engine.queue[i].costs)
"""

#logging.basicConfig(level=logging.WARN)
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.WARN, datefmt="%H:%M:%S")
print("Loading IBMQ account...")
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
print("Account loaded...")


# pdb.set_trace()
#qpu1 = TestQPU("test1", 1, "fifo")
#qpu1 = IBMQQPU("FakeAthens", "fifo")
qpu1 = IBMQQPU("ibm_nairobi", "fifo", provider=provider)

#qpu2 = TestQPU("test2", 2, "fifo")
#qpu2 = IBMQQPU("FakeAthensV2", "fifo")
qpu2 = IBMQQPU("ibm_oslo", "fifo", provider=provider)

#qpu3 = TestQPU("test3", 3, "fifo")
#qpu3 = IBMQQPU("FakeWashingtonV2", "fifo")
#qpu3 = IBMQQPU("ibm_qasm_simulator", "fifo", provider=provider)

qpus = [qpu1, qpu2]

dist_engine = Distributor(qpus, "fifo")

jobs = generate_jobs(dist_engine, 10)
for i in jobs:
    dist_engine.run_qernel(i, None)

# list_jobs(dist_engine)

print("OK")

# Wait until all the threads.
input("")
