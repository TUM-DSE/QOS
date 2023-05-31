from qos.api import QOS
from qos.types import Job
from time import sleep
import pdb
import subprocess
from qiskit.circuit.random import random_circuit
import logging
import json

# This is an sample client's code


def main():
    # pdb.set_trace()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=10)

    qos = QOS()

    newJob = Job()
    circuit = random_circuit(2, 2, measure=True)
    newJob.args["shots"] = 1000
    newJob.args["circuit"] = circuit.qasm()

    newJobId = qos.run(newJob)

    results = qos.results(newJobId)

    while results == 1:
        if results == 1:
            print("Job is still running")

        sleep(0.5)

        results = qos.results(newJobId)

    print(results)


main()
