from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer

q = QuantumRegister(2, "q")
circ = QuantumCircuit(q)
circ.h(q[0])
circ.cx(q[0], q[1])
circ.rz(0.5, q[1])
circ.measure_all()

print(circ)

test = circuit_to_dag(circ).count_ops_longest_path()

print(test)

# dag = circuit_to_dag(circ)
# dag_drawer(dag)
