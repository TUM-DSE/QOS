from pydoc import describe
import sys

sys.path.insert(0, "./../../../.")

from qstack.qernel import Qernel
from qstack.backends import TestQPU
from qstack.qos import Distributor
from qstack.types import QPUWrapper
from qiskit import QuantumCircuit, QuantumRegister

def gen_jobs(dist_engine, n_jobs:int) -> Qernel:
	empty_circuit = QuantumRegister(1)
	empty_qernel = Qernel(empty_circuit)
	
	for i in range(n_jobs):
		tmp_id = dist_engine.register_qernel(empty_qernel,None)

def list_jobs(dist_engine):
	print("There are currently:", dist_engine.job_counter, "on the queue")

	for i in range(dist_engine.job_counter):
		print("Job", i, ":", dist_engine.queue[i].costs)

qpu1 = TestQPU("test1", 1)
qpu2 = TestQPU("test2", 2)
qpu3 = TestQPU("test3", 3)

qpus = [qpu1,qpu2,qpu3]

dist_engine = Distributor(qpus, "fifo")

gen_jobs(dist_engine, 10)

list_jobs(dist_engine)



print("OK")