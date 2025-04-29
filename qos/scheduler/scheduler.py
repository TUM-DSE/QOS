from typing import Any, Dict, List
from qos.types.types import Qernel
import logging
import jsonpickle
from qos.backends.test_qpu import TestQPU
from qos.backends.ibmq import IBMQPU
from qos.types.types import Engine, Qernel
from qiskit.circuit import QuantumCircuit
from qiskit_ibm_provider import IBMProvider
import qos.database as db
#from qiskit import execute
from qiskit.transpiler import CouplingMap
import json
from settings.ibm_token import IBM_TOKEN
import pdb
from qos.tools import predict_queue_time

SHOTS = 8192
DEFAULT_FID_WEIGHT = 0.7
DEFAULT_UTIL_WEIGHT = 0

#Applying scoring formula
def compute_score(fid1, fid2, eta1, eta2, util1, util2, fid_weight, util_weight):
        try:
            score = (fid_weight*(fid2/fid1-1) - (1-fid_weight)*(eta2/eta1-1) + util_weight*(util2/util1-1))
        except ZeroDivisionError:
            fid1 += 0.00000001
            score = (fid_weight*(fid2/fid1-1) - (1-fid_weight)*(eta2/eta1-1) + util_weight*(util2/util1-1))
        #return (fid_weight*(fid2/fid1) - (1-fid_weight)*(eta2/eta1) + util_weight*(util2/
        #util1))
        return score

# From a circuit mapping copmute utlization percentage
def util_from_mapping(layout, qpu) -> float:

        circuit_qubits = len(layout)
        qpu_qubits = int(db.getQPUField(qpu, "nqbits").decode("utf-8"))

        #for i in matching:
        #    util += i[1]
        return circuit_qubits/qpu_qubits


class Scheduler(Engine):

    logger = logging.getLogger(__name__)

    def __init__(self) -> int:
        pass

    def _bestqpu_policy(self, qernels: Qernel | List[Qernel]) -> None:
        # pdb.set_trace()

        for i in qernels:
            i.match = i.matching[0]
            i.args["shots"] = SHOTS
            db.submitQernel(i)
        return
    
    def _balanced_policy(self, qernels: Qernel | List[Qernel], fid_weight=DEFAULT_FID_WEIGHT, util_weight=DEFAULT_UTIL_WEIGHT) -> None:

        #pdb.set_trace()

        if not isinstance(qernels, list):
            qernels = [qernels]
        for qernel in qernels:

            #pdb.set_trace()
            for j in range(len(qernel.matching)):
                if j >= len(qernel.matching)-1:
                    qernel.match = qernel.matching[j]
                    qernel.args["shots"] = SHOTS
                    db.submitQernel(qernel)
                    print("Submitted qernel {}".format(qernel.id))
                    break

                fid1 = 1-qernel.matching[j][2]
                fid2 = 1-qernel.matching[j+1][2]

                #local_queue2 = db.getQPUField(qernel.matching[j+1][1], "local_queue")
                earliest1 = db.QPU_earliest_free_time(qernel.matching[j][1])
                earliest2 = db.QPU_earliest_free_time(qernel.matching[j+1][1])
                eta1 = earliest1+1 if earliest1 > qernel.submit_time else qernel.submit_time+1
                eta2 = earliest2+1 if earliest2 > qernel.submit_time else qernel.submit_time+1

                #Adding 1 nanosecond to avoid division by zero, in case one of the qpus is already free, which means that the eta would be 0. 1 nanosecond is negligible

                util1 = util_from_mapping(qernel.matching[j][0], qernel.matching[j][1])
                util2 = util_from_mapping(qernel.matching[j+1][0], qernel.matching[j+1][1])
                #pdb.set_trace()
                score = compute_score(fid1, fid2, eta1, eta2, util1, util2, fid_weight, util_weight)

                #pdb.set_trace()

                if score <= 0:
                    qernel.match = qernel.matching[j]
                    qernel.args["shots"] = SHOTS
                    db.submitQernel(qernel)
                    print("Submitted qernel {}".format(qernel.id))
                    break
                else:
                    continue
        return
    
    def _lightload_balance_policy(self, new_qernel: Qernel) -> None:
        self.logger.log(10, "Running best qpu policy")
        predict_queue_time(1)
        # pdb.set_trace()
        for i in new_qernel.subqernels:
            tmpqernel = db.getQernel(i)
            tmpqernel.args["qpu"] = tmpqernel.best_qpu()
            tmpqernel.args["shots"] = 1000
            db.updateQernel(i, tmpqernel)
        return

    def run(self, qernels: Qernel|List[Qernel], policy = _balanced_policy, fid_weight=DEFAULT_FID_WEIGHT, util_weigth=DEFAULT_UTIL_WEIGHT) -> None:
        
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

        # Apply policies
        if policy == "bestqpu":
            policy = self._bestqpu_policy
            policy(all_qernels)  # Assign qpus to the subqernels
        elif policy == "balanced":
            policy = self._balanced_policy
            policy(all_qernels, fid_weight, util_weigth)

        all_qpus = db.getAllQPU()

        all_queues = []

        #print("Queues {}".format([i.local_queue for i in all_qpus]))    

        for i in all_qpus:

            print("Local queue {}".format(i.local_queue))

            #The local_queue is a list of tuple with the following information: (qernel_id, estimated execution time, submitted time, estimated waiting time, predicted fidelity)
            for j in range(len(i.local_queue)):
                for w in all_qernels:
                    if w.id == int(i.local_queue[j][0]):
                        #w.results = job_results[j]
                        w.waiting_time = i.local_queue[j][3]
                        break
            all_queues.append((i.name, i.local_queue))

        return all_queues

    def results(self) -> None:
        pass
