from typing import Any, Dict, List
from qos.types import Qernel
import logging
import jsonpickle
from qos.backends.test_qpu import TestQPU
from qos.backends.ibmq import IBMQPU
from qos.types import Engine, Qernel
from qiskit.circuit import QuantumCircuit
from qiskit_ibm_provider import IBMProvider
import qos.database as db
from qiskit import execute
from qiskit.transpiler import CouplingMap
import json
from ibm_token import IBM_TOKEN
import pdb
from qos.tools import predict_queue_time

SHOTS = 8192
DEFAULT_FID_WEIGHT = 0.7
DEFAULT_UTIL_WEIGHT = 0

def compute_score(fid1, fid2, eta1, eta2, util1, util2, fid_weight, util_weight):
        return (fid_weight*(fid2/fid1-1) - (1-fid_weight)*(eta2/eta1-1) + util_weight*(util2/util1-1))
        #return (fid_weight*(fid2/fid1) - (1-fid_weight)*(eta2/eta1) + util_weight*(util2/
        #util1))

def util_from_matching(layout, qpu) -> float:

        circuit_qubits = len(layout)
        qpu_qubits = int(db.getQPUField(qpu, "nqbits").decode("utf-8"))

        #for i in matching:
        #    util += i[1]
        return circuit_qubits/qpu_qubits

class Scheduler(Engine):

    logger = logging.getLogger(__name__)
    # policy: scheduler_policy

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
                eta1 = earliest1+1 if earliest1 > qernel.submit_time else qernel.submit_time+1/1000000000
                eta2 = earliest2+1 if earliest2 > qernel.submit_time else qernel.submit_time+1/1000000000

                #Adding 1 nanosecond to avoid division by zero, in case one of the qpus is already free, which means that the eta would be 0. 1 nanosecond is negligible

                util1 = util_from_matching(qernel.matching[j][0], qernel.matching[j][1])
                util2 = util_from_matching(qernel.matching[j+1][0], qernel.matching[j+1][1])

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

    def run(self, qernels: Qernel|List[Qernel], policy = _balanced_policy, fid_weight=DEFAULT_FID_WEIGHT, util_weigth=DEFAULT_UTIL_WEIGHT) -> None:
        
        if isinstance(qernels, list):
            root_qernels = qernels.copy()

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

        # Apply policies
        if policy == "bestqpu":
            policy = self._bestqpu_policy
            policy(all_qernels)  # Assign qpus to the subqernels
        elif policy == "balanced":
            policy = self._balanced_policy
            policy(all_qernels, fid_weight, util_weigth)

        all_qpus = db.getAllQPU()

        all_queues = []

        provider = IBMProvider(token=IBM_TOKEN)
        print("Queues {}".format([i.local_queue for i in all_qpus]))    

        for i in all_qpus:

            print("Local queue {}".format(i.local_queue))
            #Run the qernels on the local qpu on the ibm cloud
            backend = provider.get_backend(i.alias)
            #local_queue_circuits =  [qernels[int(j[0])].circuit for j in i.local_queue]
            local_queue_circuits =  []

            #The local_queue is a list of tuple with the following information: (qernel_id, estimated execution time, submitted time, estimated waiting time, predicted fidelity)

            # Imagine you have the following qernels
            # qernel0 (main qernel) -> qernel1 (sub), qernel2 (sub)
            # qernel4 (main qernel) -> qernel5 (sub), qernel6 (sub)
            # qernel7 (main qernel) -> qernel8 (sub), qernel9 (sub)

            #The all_qernels list will have: q1, q2, q5, q6, q8, q9
            #After the scheduling the local queue of a qpu could have just qernels: q5, q8
            #Now I need to get the circuit of these qernel to submit to ibm
            #Also it need to be in the same order as the local queue because after I get the results I need to know which result it belongs to which qernel

            for j in i.local_queue:
                for w in all_qernels:
                    if w.id == int(j[0]):
                        local_queue_circuits.append(w.circuit)
                        break

            #mappings = [qernels[int(j[0])].match[0] for j in i.local_queue]

            results = []
            pdb.set_trace()

            #for j in range(len(local_queue_circuits)):
            print("Running batch on qpu {}".format(i.name))
            #circuit = local_queue_circuits[j]
            #remapping = [[i,mappings[j][i]] for i in range(len(mappings[j]))]
            #coupling_map = CouplingMap(remapping)
            job_results = execute(local_queue_circuits, backend=backend, shots=8192).result().get_counts()
                #job = execute(circuit, backend=backend, shots=8192)
                #results.append(job.result().get_counts())

            for j in range(len(i.local_queue)):
                for w in all_qernels:
                    if w.id == int(i.local_queue[j][0]):
                        w.results = job_results[j]
                        w.waiting_time = i.local_queue[j][3]
                        break
            
            results.append(job_results)
            
            all_queues.append([i.name, i.local_queue, results])
            #print('Final queue for qpu {}:'.format(i.name))
            #dist.append((i.name, len(i.local_queue)))
            #for j in i.local_queue:
            #       print('Submitted circuit {} at {}, eta: {}'.format(j[0], j[2], j[1]))

            #print('---------------------------------\n')

        #db.setQernelField(qernels.id, "status", "DONE")

        # stat = db.getQernelField(qernel.id, "status").decode("utf-8")

        return all_queues

    # This policy follows the following rules:
    # 1. If the qernel has more than one subqernel, means that it was merged and then use the assigned QPU

    def results(self) -> None:
        pass

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
