from typing import List
import qos.database as db
from qos.types import Backend, Qernel
import pdb
import logging
from qos.engines.scheduler import Scheduler
import queue
from multiprocessing import Process
from time import sleep
import os

pipe_name = "multiprog_fifo.pipe"

def check_layout_overlap(layout1: List, layout2: List) -> bool:
    for i in range(0, len(layout1)):
        if layout1[i] in layout2:
            return True
    return False


class Multiprogrammer:
    def __init__(self) -> None:
        Process(target=self.window_monitor).start()
        sleep(2)
        return

    #def submit(self, qernel: Qernel):
    #
    #    # Here the multiprogramming engine would do its qernel
    #
    #    self.multiprogram(qernel, self._base_policy)
    #
    #    sched = Scheduler()
    #    sched.submit(qernel, sched._lightload_balance_policy)
    #
    #    return 0

    def window_monitor(self):
        # pdb.set_trace()
        os.mkfifo(pipe_name)
        openfifo = open(pipe_name, "r")

        while True:
            print("Waiting for message")
            line = openfifo.readline()
            if not line:
                continue
            else:
                print(
                    line + "received message"
                )
                # ? This probably can just be the qernel id, and then we can get the qernel from the database?
                qernel = db.getQernel(int(line))
                self.multiprogram(qernel, self._restrict_policy)

    def multiprogram(self, qernel: Qernel, merge_policy):
        this = merge_policy(qernel, 0.1)
        return this

    # Merging policies:

    # 1. Restrist policy: Only merge if the best QPU for two circuits is the same and their best layouts dont overlap
    #   Start by considering one of the new circuits with the oldest circuits on the window
    #   After considering merging the new circuits with each one on the window move the window the number of circuits as the number of circuits that couldnt be merged
    #   Example: The window has 5 circuits E, D, C, B, A and the new circuits are F, G, H.
    #   1. Consider merging F or G or H with E, D, C, B, A, by this order. Lets consider that F could be merged with A
    #   2. The new circuits left are G and H, move the window by two circuits, this is because the new circuits need to enter the window and the size of the window is fixed
    def _restrict_policy(self, new_qernel: Qernel, error_limit: float) -> None:
        # self.logger.log(10, "Running Restrict policy")
        window = db.currentWindow()

        for i in window[::-1]:
            # ! This is not supposed to go through the subqernels, just the matchings, fix this
            for j in new_qernel.subqernels:
                if j.best_qpu == i.best_qpu and not check_layout_overlap(
                    j.best_layout(), i.best_layout()
                ):
                    print("Multiprogramming match found!!")
                else:
                    print("No match found")

        return 0

    def _base_policy(self, newQernel: Qernel, error_limit: float) -> None:
        # self.logger.log(10, "Running Base policy")

        logger = logging.getLogger(__name__)
        logging.basicConfig(level=10)

        window = db.currentWindow()

        for j in newQernel.subqernels:
            for i in window[::-1]:
                for x in i.matching:
                    print(x)
                    if not check_layout_overlap(j.best_layout(), i.best_layout()):
                        logger.log(
                            10, "Possible match found, checking score threshold."
                        )

                        # If the average of the scores is above the threshold fidelidy, then merge
                        if ((j.best_score + i.best_score) / 2) > error_limit:
                            logger.log(10, "Multiprogramming match found!")
                            return 0

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
