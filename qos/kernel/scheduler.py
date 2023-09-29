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

class Scheduler(Engine):

    logger = logging.getLogger(__name__)
    # policy: scheduler_policy

    def __init__(self) -> int:
        pass

    def _bestqpu_policy(qernels: Qernel | List[Qernel]) -> None:
        # pdb.set_trace()

        for i in qernels:
            i.match = i.matching[0]
            i.args["shots"] = 8192*100
            db.submitQernel(i)
        return


    def run(self, qernels: Qernel|List[Qernel], policy = _bestqpu_policy) -> None:

        policy(qernels)  # Assign qpus to the subqernels
        
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

        for i in all_qpus:
            print('Final queue for qpu {}:'.format(i.name))
            dist.append((i.name, len(i.local_queue)))
            for j in i.local_queue:
                   print('Submitted circuit {} at {}, eta: {}'.format(j[0], j[2], j[1]))

            print('---------------------------------\n')

        #db.setQernelField(qernels.id, "status", "DONE")

        # stat = db.getQernelField(qernel.id, "status").decode("utf-8")

        return dist

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
