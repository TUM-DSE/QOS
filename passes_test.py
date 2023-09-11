from qos.distributed_transpiler.analyser import *
from qiskit.circuit.random import random_circuit
from qos.types import Qernel


def test_analyses_passes(qernel: Qernel) -> None:
    basic_pass = BasicAnalysisPass()
    basic_pass.run(qernel)

    supermarq_feature_pass = SupermarqFeaturesAnalysisPass()
    supermarq_feature_pass.run(qernel)

    dg = DependencyGraphFromDAGPass()
    dg.run(qernel)


def main():
    qc = random_circuit(5, 5, max_operands=2, measure=True)
    print(qc)

    qernel = Qernel(qc)

    test_analyses_passes(qernel)
    print(qernel.get_metadata())

main()