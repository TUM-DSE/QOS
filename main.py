from qos.api import QOS
from qos.types import Job, QCircuit
from time import sleep
import pdb
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
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

    # This is a random circuit, the results should be counts split equally on the values 0000, 0011, 1100, 1111
    qreg_q = QuantumRegister(4, "q")
    circuit = QuantumCircuit(qreg_q)
    circuit.h(qreg_q[0])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.h(qreg_q[2])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.measure_all()

    logger.log(10, "Submitting circuit")

    newJobId = qos.run(circuit)

    logger.log(10, "Circuit submitted")

    results = qos.results(newJobId)

    logger.log(10, "Results tentative fetch")

    while results == 1:
        if results == 1:
            print("Job is still running")

        sleep(100)

        results = qos.results(newJobId)
        logger.log(10, "Results tentative fetch")

    print(results)


main()
