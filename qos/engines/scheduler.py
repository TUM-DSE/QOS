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

    def run(self, qernel: Qernel, policy) -> None:

        self.logger.log(10, "Got new qernel to be scheduled")
        # results = {"test": 43}

        policy(qernel)  # Assign qpus to the subqernels

        for i in qernel.subqernels:

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

        # This is not supposed to be like this, this should run on a new process and update the database when the qernel is done in the cloud
        db.setQernelField(qernel.id, "status", "DONE")

        # stat = db.getQernelField(qernel.id, "status").decode("utf-8")

        return 0

    def _bestqpu_policy(self, new_qernel: Qernel) -> None:
        self.logger.log(10, "Running best qpu policy")
        # pdb.set_trace()
        for i in new_qernel.subqernels:
            tmpqernel = db.getQernel(i)
            tmpqernel.args["qpu"] = tmpqernel.best_qpu()
            tmpqernel.args["shots"] = 1000
            db.updateQernel(i, tmpqernel)
        return

    # This policy follows the following rules:
    # 1. If the qernel has more than one subqernel, means that it was merged and then use the assigned QPU
    # 2.

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
