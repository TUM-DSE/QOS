from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer
from qos.dag import DAG
import pickle

qreg_q = QuantumRegister(4, "q")
circuit = QuantumCircuit(qreg_q)
circuit.h(qreg_q[0])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.h(qreg_q[2])
circuit.cx(qreg_q[2], qreg_q[3])
circuit.measure_all()

print(circuit)

dag = DAG(circuit)
this = pickle.dumps(dag)
print(this)

new = pickle.loads(this)
#dag.draw("dag.png")
this = new.to_circuit()
print(this)

# dag = circuit_to_dag(circ)
# dag_drawer(dag)
