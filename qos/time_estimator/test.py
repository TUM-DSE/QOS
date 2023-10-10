from regression_estimator import RegressionEstimator
from qiskit import QuantumCircuit


def main():
    #Random circuit
    circ = random_circuit(8, 5, max_operands=2, measure=True)
