from typing import List
import qos.database as db
from qos.types import Backend, QCircuit, Job
import pdb
from qos.engines.scheduler import Scheduler


def check_layout_overlap(layout1: List, layout2: List) -> bool:
    for i in range(0, len(layout1)):
        if layout1[i] in layout2:
            return True
    return False


class Multiprogrammer:
    def __init__(self) -> None:
        pass

    def submit(self, job: Job):

        # Here the multiprogramming engine would do its job

        print(self.multiprogram(job, self._restrict_policy))
        print("-------------------------")

        sched = Scheduler()
        sched.submit(job, sched._bestqpu_policy)

        return 0

    def multiprogram(self, job: Job, merge_policy):
        this = merge_policy(job)
        return this

    # Merging policies:
    # Restrist policy: Only merge if the best QPU for two circuits is the same and their best layouts dont overlap
    #   Start by considering one of the new circuits with the oldest circuits on the window
    #   After considering merging the new circuits with each one on the window move the window the number of circuits as the number of circuits that couldnt be merged
    #   Example: The window has 5 circuits E, D, C, B, A and the new circuits are F, G, H.
    #   1. Consider merging F or G or H with E, D, C, B, A, by this order. Lets consider that F could be merged with A
    #   2. The new circuits left are G and H, move the window by two circuits, this is because the new circuits need to enter the window and the size of the window is fixed
    def _restrict_policy(self, new_job: Job) -> None:
        # self.logger.log(10, "Running Restrict policy")
        window = db.currentWindow()

        for i in window[::-1]:
            for j in new_job.subjobs:
                if j.best_qpu == i.best_qpu and not check_layout_overlap(
                    j.best_layout(), i.best_layout()
                ):
                    print("Multiprogramming match found!!")
                else:
                    print("No match found")

        return 0

        # for i in window[0,-1]():


#  def __merge_qernels(self, qc1: QCircuit, q2: QCircuit) -> QCircuit:
# toReturn = Qernel(q1.num_qubits + q2.num_qubits, q1.num_clbits + q2.num_clbits)
# qubits1 = [*range(0, q1.num_qubits)]
# clbits1 = [*range(0, q1.num_clbits)]
# qubits2 = [*range(q1.num_qubits, q1.num_qubits + q2.num_qubits)]
# clbits2 = [*range(q1.num_clbits, q1.num_clbits + q2.num_clbits)]

# toReturn.compose(q1, qubits=qubits1, clbits=clbits1, inplace=True)
# toReturn.compose(q2, qubits=qubits2, clbits=clbits2, inplace=True)
# return toReturn
