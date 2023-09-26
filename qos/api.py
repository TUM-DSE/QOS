from typing import Any, Dict, List
from qos.types import Qernel
from .distributed_transpiler.optimiser import Optimiser
import qos.database as database
import qos.distributed_transpiler.analyser as Analyser
import redis
import logging
#from qos.types import QCircuit
from qiskit import QuantumCircuit
import qos.tools
from multiprocessing import Process
import pdb
from qos.engines.multiprogrammer import pipe_name as multiprog_pipe_name
from qos.engines.multiprogrammer import Multiprogrammer
import qos.distributed_transpiler.run as DT
import os
from qos.tools import debugPrint
from .dag import DAG
import yaml

class QOS:
    """Main API that will be exposed to the user"""

    optimiser: Optimiser
    logger = logging.getLogger(__name__)
    workers: List[
        Process
    ]  # I think we wont be needing this, the API just issues the qernel and does nothing else with it

    def __init__(self) -> None:
        # self.db = database()
        self.logger = logging.getLogger(__name__)

        # Ignoring certain loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("qiskit").setLevel(logging.WARNING)
        logging.getLogger("stevedore").setLevel(logging.WARNING)

        self.logger.log(10, "Checking if multiprogrammer is running")
        # pdb.set_trace()

        #We before starting the system again we need to delete the pipe file, otherwise         print(configs.get("passes"))the Multiprogrammer
        #wont open the file for reading and the Matcher will stall on the open for writting
        if not os.path.exists(multiprog_pipe_name):
            self.logger.log(10, "Multiprogrammer not running, starting it")
            # Create a FIFO pipe
            Multiprogrammer()

        self.optimiser = Optimiser()
        self.logger.log(10, "QOS API initialized")
        self.workers = []

    def run(self, circuit: Any) -> int:

        #newQC = QCircuit()
        newQernel = Qernel()

        # * Here the circuits from different providers are converted into DAGS
        if type(circuit) == QuantumCircuit:
            newQernel.provider = "qiskit"
            newQernel.circuit = DAG(circuit)
        else:
            print("Circuit provider not supported yet")
            exit(1)

        # Adds the qernel to the database
        qernelId = database.addQernel(newQernel)
        newQernel.id = qernelId

        self.logger.log(10, "New qernel added to the database")

        self.workers.append(Process(target=self.worker_start, args=(newQernel,)))

        self.logger.log(10, "Opening new process, sumbitting QC to analyser")
        self.workers[-1].start()
        self.workers[-1].join() # ! I think this is not supposed to join, it just starts and it will exit on its own, we need to check this

        return qernelId

    def results(self, qernelId: int) -> None:

        stat = database.getQernelField(qernelId, "status")

        if stat == b"DONE":
            qernel = database.getQernel(qernelId)
            # Probably now we would process the results from the subqernels and return the final qernel result
            # pdb.set_trace()
            # tmpqernel = database.getQernel(qernel.subqernels[0])
            results = database.getQernelField(qernel.subqernels[0], "results").decode()
            return results
        else:
            return 1

    def worker_start(self, qernel: Qernel) -> None:
        # pdb.set_trace()

        dist_transpiler = DT.DistributedTranspiler(qernel, 5)
        qernel = dist_transpiler.run()
        
        #-------
        

        qernel = Analyser.analyse(qernel)
        self.optimiser.submit(qernel)
        
