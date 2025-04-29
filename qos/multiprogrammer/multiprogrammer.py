from qos.types.types import Engine
from typing import Any, Dict, List
import qos.database as db
from qos.types.types import Backend, Qernel
import pdb
import logging
from qos.scheduler import Scheduler
from qos.tools import check_layout_overlap, size_overflow, bundle_qernels
import queue
from multiprocessing import Process
from time import sleep
import os
import numpy as np

pipe_name = "multiprog_fifo.pipe"

class Multiprogrammer(Engine):

    timeout = 5
    window_size = 5
    
    def __init__(self) -> None:
        #Process(target=self.window_monitor).start()
        #sleep(2)
        #return
        self.queue = []
        self.done_queue = []

    def get_matching_score(self, q1: Qernel, q2: Qernel, weighted: bool = False, weights: List[float] = []) -> float:
        score = 0

        depthDiff = self.depthComparison(q1, q2)
        entanglementDiff = self.entanglementComparison(q1, q2)
        measurementDiff = self.measurementComparison(q1, q2)
        parallelismDiff = self.parallelismComparison(q1, q2)

        if weighted and sum(weights) > 0:            
            score = weights[0] * depthDiff + weights[1] * entanglementDiff + weights[2] * measurementDiff + weights[3] * parallelismDiff
        else:
            score = (depthDiff + entanglementDiff + measurementDiff + parallelismDiff) / 4
    
        return score
    

    def depthComparison(self, q1: Qernel, q2: Qernel) -> float:
        metadata_1 = q1.get_metadata()
        metadata_2 = q2.get_metadata()

        depthDiff = np.abs(metadata_1["depth"] - metadata_2["depth"])
        
        k = 0.05

        depthFactor = np.exp(-k * depthDiff)
                   
        return depthFactor
    
    def entanglementComparison(self, q1: Qernel, q2: Qernel) -> float:
        metadata_1 = q1.get_metadata()
        metadata_2 = q2.get_metadata()

        entanglement_result = (1 - metadata_1["entanglement_ratio"]) * (1 - metadata_2["entanglement_ratio"])

        return entanglement_result
    
    def measurementComparison(self, q1: Qernel, q2: Qernel) -> float:
        metadata_1 = q1.get_metadata()
        metadata_2 = q2.get_metadata()

        measurement_result = (1 - metadata_1["measurement"]) * (1 - metadata_2["measurement"])

        return measurement_result
    
    def parallelismComparison(self, q1: Qernel, q2: Qernel) -> float:
        metadata_1 = q1.get_metadata()
        metadata_2 = q2.get_metadata()

        parallelism_result = (1 - metadata_1["parallelism"]) * (1 - metadata_2["parallelism"])

        return parallelism_result

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

    #def window_monitor(self):
    #    # pdb.set_trace()
    #    #os.mkfifo(pipe_name)
    #    #openfifo = open(pipe_name, "r")

    #    while True:
    #        print("Waiting for message")
    #        line = openfifo.readline()
    #        if not line:
    #            continue
    #        else:
    #            print(
    #                line + "received message"
    #            )
    #            # ? This probably can just be the qernel id, and then we can get the qernel from the database?
    #            qernel = db.getQernel(int(line))
    #            self.multiprogram(qernel, self._restrict_policy)

    def run(self, qernels: List[Qernel] | Qernel, merge_policy='restrict') -> List[Qernel]:


        if merge_policy == 'restrict':
            merge_policy = self._restrict_policy
        else:
            print("Merge policy not supported yet")
            exit(1)

        #Its much easier to apply the merge policy if the subqernels are on the same list, just iterate through the list, or we can always pass around the root qernels, its much more clean but might take more time to look around for subqernels and subsubqernels
        ''' 
        if isinstance(qernels, list):
            all_qernels = []

            for q in qernels:
                sub_qernels = q.get_subqernels()
                if sub_qernels != []:
                    for subq in sub_qernels:
                        subsub_qernels = subq.get_subqernels()
                        if subsub_qernels != []:
                            for subsubq in subsub_qernels:
                                all_qernels.append(subsubq)
                        else:
                            all_qernels.append(subq)

                else:
                    all_qernels.append(q)
            
                #if len(self.queue) > self.window_size:
                #    self.done_queue.append(self.queue.pop())


            merge_policy(all_qernels, 0.1)
            return self.done_queue

        else:
            this = merge_policy(qernels, 0.1)
            return this
        
        return qernels
        '''
        #The restrict policy doesnt work if the sub and subsubqernels are on the same list
        #merge_policy(qernels, 0.1)

        return qernels #<--- Comment this out
        
        # Copy paste here \/ your multiprogrammer code
        # It should return the final queue of qernels to be scheduled






        # return final_queue
        # ---------------------
        
        
        #return self.done_queue        
    
    def results(self) -> None:
        pass
  

    def test_policy(self, qernels: List[Qernel]) -> None:
        for q in qernels:
            return q

     # Merging policies:

    # 1. Restrist policy: Only merge if the best QPU for two circuits is the same and their best layouts dont overlap
    #   Start by considering one of the new circuits with the oldest circuits on the window
    #   After considering merging the new circuits with each one on the window move the window the number of circuits as the number of circuits that couldnt be merged
    #   Example: The window has 5 circuits E, D, C, B, A and the new circuits are F, G, H.
    #   1. Consider merging F or G or H with E, D, C, B, A, by this order. Lets consider that F could be merged with A
    #   2. The new circuits left are G and H, move the window by two circuits, this is because the new circuits need to enter the window and the size of the window is fixed
    
    def _restrict_policy(self, new_qernels: Qernel | List[Qernel] , error_limit: float, matching_cycles=1, max_bundle=2) -> None:
        # self.logger.log(10, "Running Restrict policy")
        #window = db.currentWindow()

        # This is whole algorithm is very very unclean, but it works for now
        # This compares matching 0 of the incoming qernel with matching 0 of a qernel on the queue, the 1 to 0 then 0 to 1 and so on
        cycles = [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,2), (2,2), (3,0), (0,3), (3,1), (1,3), (3,2), (2,3), (3,3)]

        if isinstance(new_qernels, Qernel):
            #TODO
            return

        #There is a caveat here, if a qernel has been merged the matching will always be the same, so it might make some repeated checks

        next = 0

        for q in new_qernels:
            for cycle in range(0, matching_cycles):
                if next == 1:
                    next = 0
                    break
                if self.queue == []:
                    break
                for queue in self.queue[::-1]:
                    incoming_qpu = q.matching[cycles[cycle][0]][1]
                    queue_qpu = queue.matching[cycles[cycle][1]][1] if queue.match == None else queue.match[1]
                    incoming_map = q.matching[cycles[cycle][0]][0]
                    queue_map = queue.matching[cycles[cycle][1]][0] if queue.match == None else queue.match[0]
                    incoming_fid = q.matching[cycles[cycle][0]][2]
                    queue_fid = queue.matching[cycles[cycle][1]][2] if queue.match == None else queue.match[2]
                    #The first element of the matching is the ideal mapping, the second is the qpu and the third is the estimated fidelity
                    if incoming_qpu == queue_qpu and not check_layout_overlap(incoming_map, queue_map) and not size_overflow(q, queue, incoming_qpu) and not (((incoming_fid + queue_fid) / 2) > error_limit):
                                    
                        print("Multiprogramming match found!")
                        tmp = bundle_qernels(q, queue, (incoming_map, incoming_qpu, incoming_fid))
                                    
                        if tmp == 0:
                        # This is the case that the qernel on the queue already was a bundled qernel and the new qernel was just added to the bundle, nothing more to do.
                            tmp = queue
                        
                        else:
                        #This is the case that the qernel on the queue was a normal qernel and a new bundled qernel was created this new qernel is going to substitute the original qernel on the queue
                            self.queue.remove(queue)
                            self.queue.insert(0,tmp)
                                        
                        if len(tmp.src_qernels) >= max_bundle:
                        #If the bundled qernel already has the max bundle size, then it jumps the queue and is scheduled
                            self.done_queue.insert(0,tmp)
                            self.queue.remove(tmp)

                        next = 1
                        break

            self.queue.insert(0,q)

            if len(self.queue) > self.window_size:
                self.done_queue.insert(0,self.queue.pop())

        self.done_queue = self.queue + self.done_queue
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