from typing import Any, Dict, List
from qos.types import Qernel
import logging
import jsonpickle
from qos.backends.test_qpu import TestQPU
from qos.backends.ibmq import IBMQPU
from qos.types import Engine, Qernel
from qiskit.circuit import QuantumCircuit
import qos.database as db
import json
import pdb
from qos.tools import predict_queue_time

SHOTS_MULTIPLIER = 1
FID_WEIGHT = 0.75
UTIL_WEIGHT = 0

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
            i.args["shots"] = 8192*SHOTS_MULTIPLIER
            db.submitQernel(i)
        return
    
    def _balanced_policy(self, qernels: Qernel | List[Qernel]) -> None:

        if not isinstance(qernels, list):
            qernels = [qernels]
        for qernel in qernels:
            current_best = qernel.matching[0]

            #pdb.set_trace()
            for j in range(len(qernel.matching)):
                if j >= len(qernel.matching)-1:
                    qernel.match = qernel.matching[j]
                    qernel.args["shots"] = 8192*SHOTS_MULTIPLIER
                    db.submitQernel(qernel)
                    break

                fid1 = 1-qernel.matching[j][2]
                fid2 = 1-qernel.matching[j+1][2]

                #local_queue2 = db.getQPUField(qernel.matching[j+1][1], "local_queue")
                earliest1 = db.QPU_earliest_free_time(qernel.matching[j][1])
                earliest2 = db.QPU_earliest_free_time(qernel.matching[j+1][1])
                eta1 = earliest1+1 if earliest1 > qernel.submit_time else qernel.submit_time+1
                eta2 = earliest2+1 if earliest2 > qernel.submit_time else qernel.submit_time+1

                #Adding 1 nanosecond to avoid division by zero, in case one of the qpus is already free, which means that the eta would be 0. 1 nanosecond is negligible

                util1 = util_from_matching(qernel.matching[j][0], qernel.matching[j][1])
                util2 = util_from_matching(qernel.matching[j+1][0], qernel.matching[j+1][1])

                score = compute_score(fid1, fid2, eta1, eta2, util1, util2, FID_WEIGHT, UTIL_WEIGHT)

                #pdb.set_trace()

                if score <= 0:
                    qernel.match = qernel.matching[j]
                    qernel.args["shots"] = 8192*SHOTS_MULTIPLIER
                    db.submitQernel(qernel)
                    break
                else:
                    continue
        return
    

    def run(self, qernels: Qernel|List[Qernel], policy = _balanced_policy) -> None:

        if policy == "bestqpu":
            policy = self._bestqpu_policy
            policy(qernels)  # Assign qpus to the subqernels
        elif policy == "balanced":
            policy = self._balanced_policy
            policy(qernels)
        
        '''
        for i in qernels.subqernels:

            tmpqernel = db.getQernel(i)
            tmpqernel.qpu = db.getQPU_fromname(tmpqernel.args[b"qpu"].decode())

            if tmpqernel.qpu.provider == "test":
                qpu = TestQPU()
                results = qpu.run()
            elif tmpqernel.qpu.provider == "ibm":
                qpu = IBMQPU()
                circuit = QuantumCircuit.from_qasm_str(tmpqernel.circuit.decode())
                # print(tmpqernel.qpu.name)
                # print(circuit)
                # pdb.set_trace()
                trans_circuit = qpu.transpile(circuit, tmpqernel.qpu.name)
                results = qpu.run(
                    trans_circuit, tmpqernel.qpu.name, tmpqernel.shots
                ).get_counts()

            # Here the scheduler would do its qernel

            self.logger.log(10, "Got results from qpu, updating")

            db.setQernelField(tmpqernel.id, "status", "DONE")
            db.setQernelField(tmpqernel.id, "results", jsonpickle.encode(results))
        '''
        all_qpus = db.getAllQPU()

        #pdb.set_trace()

        dist = []

        all_queues = []

        for i in all_qpus:
            all_queues.append([i.name, i.local_queue])
            #print('Final queue for qpu {}:'.format(i.name))
            #dist.append((i.name, len(i.local_queue)))
            #for j in i.local_queue:
            #       print('Submitted circuit {} at {}, eta: {}'.format(j[0], j[2], j[1]))
#
            #print('---------------------------------\n')

        #db.setQernelField(qernels.id, "status", "DONE")

        # stat = db.getQernelField(qernel.id, "status").decode("utf-8")

        return all_queues

    # This policy follows the following rules:
    # 1. If the qernel has more than one subqernel, means that it was merged and then use the assigned QPU
    # 2.

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
