from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer
from qos.dag import DAG
import pickle
import redis
from qos.backends.types import QPU
import qos.database as db
from qos.types import Qernel
import pdb
from qos.database import submitQernel

qreg_q = QuantumRegister(4, "q")
circuit = QuantumCircuit(qreg_q)
circuit.h(qreg_q[0])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.h(qreg_q[2])
circuit.cx(qreg_q[2], qreg_q[3])
circuit.measure_all()

# dag = circuit_to_dag(circ)
# dag_drawer(dag)
#pdb.set_trace()


def main():
    this = Qernel(circuit)
    this.args["shots"] = 1000
    this.provider = "qiskit"

    #pdb.set_trace()

    #backend = db.getQPU_fromname('fake_cairo')
    this = db.submitQernel('fake_cairo', this)

    print(this)

    print(circuit)


main()