import pdb
import sys
import logging

sys.path.insert(0, "./../../../.")

from qstack.backends._ibm import IBMQQPU
from qstack.qos.distributor import Distributor
from qiskit.circuit.random.utils import random_circuit
from qiskit import IBMQ

def generate_jobs(dist_engine, n_jobs: int):
    circuit = random_circuit(num_qubits=3, depth=2)
    jobs = n_jobs * [circuit]
    return jobs


logging.basicConfig(format='[Scheduler] - %(message)s', level=42)

print("Loading IBMQ account and provider...")

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')

print("Account and provider loaded...")

qpu1 = IBMQQPU("ibm_nairobi", "fifo", provider=provider)
qpu2 = IBMQQPU("ibm_oslo", "fifo", provider=provider)

dist_engine = Distributor([qpu1, qpu2], "fifo")

jobs = generate_jobs(dist_engine, 4)

for i in jobs:
    dist_engine.run_qernel(i, None)

print("-----No more jobs to register-----")
