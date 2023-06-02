from typing import List
import qos.database as db
from qos.types import Backend, QCircuit, Job

from qos.engines.scheduler import Scheduler


class Multiprogrammer:
    def __init__(self) -> None:
        pass

    def submit(self, job: Job):

        # Here the multiprogramming engine would do its job

        sched = Scheduler()
        sched.submit(job)

        return 0


#  def __merge_qernels(self, qc1: QCircuit, q2: QCircuit) -> QCircuit:
# toReturn = Qernel(q1.num_qubits + q2.num_qubits, q1.num_clbits + q2.num_clbits)
# qubits1 = [*range(0, q1.num_qubits)]
# clbits1 = [*range(0, q1.num_clbits)]
# qubits2 = [*range(q1.num_qubits, q1.num_qubits + q2.num_qubits)]
# clbits2 = [*range(q1.num_clbits, q1.num_clbits + q2.num_clbits)]

# toReturn.compose(q1, qubits=qubits1, clbits=clbits1, inplace=True)
# toReturn.compose(q2, qubits=qubits2, clbits=clbits2, inplace=True)
# return toReturn
