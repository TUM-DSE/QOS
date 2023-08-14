from qos.api import QOS
from qos.types import Qernel
from typing import List
from time import sleep
import pdb
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import subprocess
from qiskit.circuit.random import random_circuit
import logging
from qos.database import getQPU
import json
from multiprocessing import Process
from qos.dag import DAG
from qos.tools import average_gate_times, qpuProperties, estimate_execution_time

# This is an sample client's code


def client1():
    qos = QOS()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=10)

    qreg_q = QuantumRegister(4, "q")
    circuit = QuantumCircuit(qreg_q)
    circuit.h(qreg_q[0])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.h(qreg_q[2])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.measure_all()

    logger.log(10, "Submitting circuit")
    print(circuit)

    qc_dag = DAG.from_circuit(circuit)
    DAG.draw(qc_dag, filename="qc_dag.png")
    
    dag_dag = DAG.from_string(qc_dag.to_string)
    DAG.draw(dag_dag, filename="dag_dag.png")

    newQernelId = qos.run(circuit)

    logger.log(10, "Circuit submitted")

    results = qos.results(newQernelId)

    # logger.log(10, "Results tentative fetch")

    while results == 1:
        if results != 1:
            logger.log(10, "Qernel 1 finished")
            print(results)

        sleep(10)

        results = qos.results(newQernelId)

    logger.log(10, "Circuit results finished")
    # logger.log(10, "Results tentative fetch")


def client2():
    qos = QOS()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=10)

    qreg_q = QuantumRegister(3, "q")
    circuit = QuantumCircuit(qreg_q)
    circuit.h(qreg_q[1])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.h(qreg_q[2])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.measure_all()

    logger.log(10, "Submitting circuit")

    newQernelId = qos.run(circuit)

    logger.log(10, "Circuit submitted")

    results = qos.results(newQernelId)

    # logger.log(10, "Results tentative fetch")

    while results == 1:
        if results != 1:
            logger.log(10, "Qernel finished")
            print(results)

        sleep(10)

        results = qos.results(newQernelId)


def client3():
    qos = QOS()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=10)

    qreg_q = QuantumRegister(3, "q")
    circuit = QuantumCircuit(qreg_q)
    circuit.h(qreg_q[1])
    circuit.h(qreg_q[2])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.measure_all()

    logger.log(10, "Submitting circuit")

    newQernelId = qos.run(circuit)

    logger.log(10, "Circuit submitted")

    results = qos.results(newQernelId)

    # logger.log(10, "Results tentative fetch")

    while results == 1:
        if results != 1:
            logger.log(10, "Qernel finished")
            print(results)

        sleep(10)

        results = qos.results(newQernelId)


def client4():

    qos = QOS()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=10)

    qreg_q = QuantumRegister(4, "q")
    circuit = QuantumCircuit(qreg_q)
    circuit.h(qreg_q[0])
    circuit.h(qreg_q[3])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.measure_all()

    logger.log(10, "Submitting circuit")

    newQernelId = qos.run(circuit)

    logger.log(10, "Circuit submitted")

    results = qos.results(newQernelId)

    # logger.log(10, "Results tentative fetch")

    while results == 1:
        if results != 1:
            logger.log(10, "Qernel finished")
            print(results)

        sleep(10)

        results = qos.results(newQernelId)


def client5():
    qos = QOS()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=10)

    qreg_q = QuantumRegister(5, "q")
    circuit = QuantumCircuit(qreg_q)
    circuit.h(qreg_q[0])
    circuit.h(qreg_q[2])
    circuit.h(qreg_q[4])
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cx(qreg_q[2], qreg_q[1])
    circuit.cx(qreg_q[1], qreg_q[4])
    circuit.measure_all()

    logger.log(10, "Submitting circuit")

    newQernelId = qos.run(circuit)

    logger.log(10, "Circuit submitted")

    results = qos.results(newQernelId)

    logger.log(10, "Results tentative fetch")

    while results == 1:
        if results != 1:
            logger.log(10, "Qernel finished")
            print(results)

        sleep(10)

        results = qos.results(newQernelId)


def main():
    # pdb.set_trace()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=10)

    clients: List[Process] = []

    qreg_q = QuantumRegister(5, "q")
    circuit = QuantumCircuit(qreg_q)
    circuit.h(qreg_q[0])
    circuit.h(qreg_q[2])
    circuit.h(qreg_q[4])
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cx(qreg_q[2], qreg_q[1])
    circuit.cx(qreg_q[1], qreg_q[4])
    circuit.measure_all()

    circ = circuit.qasm()

    # qpu = getQPU(7)

    # this = average_gate_times(qpuProperties(7))
    # pdb.set_trace()
    # estimate_execution_time(circ, this, qpu)

    clients.append(Process(target=client1))
    # clients.append(Process(target=client2))
    # clients.append(Process(target=client3))
    # clients.append(Process(target=client4))
    # clients.append(Process(target=client5))

    for i, client in enumerate(clients):
        logging.info("Starting client " + str(i))
        client.start()

    for i, client in enumerate(clients):
        logging.info("Joining client " + str(i))
        client.join()


main()
