from pydoc import describe
import sys

sys.path.insert(0, "./../../../.")
from qiskit import QuantumCircuit, QuantumRegister

from qstack.qernel.qernel import Qernel
from qstack.backends.test_qpu import TestQPU
from qstack.qos.distributor import Distributor
#from qstack.types import

def generate_jobs(dist_engine, n_jobs:int):
	empty_circuit = QuantumRegister(1)
	empty_qernel = Qernel(empty_circuit)
	jobs = n_jobs*[empty_qernel]
	return jobs

'''
def list_jobs(dist_engine):
	print("There are currently:", dist_engine.job_counter, "jobs on the queue")

	#for i in range(dist_engine.job_counter):
	#	print("Job", i, ":", dist_engine.queue[i].costs)
'''

qpu1 = TestQPU("test1", 1, "fifo")
qpu2 = TestQPU("test2", 2, "fifo")
qpu3 = TestQPU("test3", 3, "fifo")

qpus = [qpu1,qpu2,qpu3]

dist_engine = Distributor(qpus, "fifo")

jobs = generate_jobs(dist_engine, 10)

for i in jobs:
	dist_engine.register_qernel(i,None)

#list_jobs(dist_engine)

print("OK")