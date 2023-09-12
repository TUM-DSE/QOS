from qos.distributed_transpiler.analyser import *
from qos.distributed_transpiler.optimiser import *
from qiskit.circuit.random import random_circuit
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import TwoLocal
from qos.types import Qernel
from qos.virtualizer.virtualizer import GateVirtualizer
from qvm.qvm.virtual_circuit import generate_instantiations


def test_analyses_passes(qernel: Qernel) -> None:
    basic_pass = BasicAnalysisPass()
    basic_pass.run(qernel)

    supermarq_feature_pass = SupermarqFeaturesAnalysisPass()
    supermarq_feature_pass.run(qernel)

    dg = DependencyGraphFromDAGPass()
    dg.run(qernel)


def test_transformation_passes(qernel: Qernel) -> Qernel:
    bisection_pass = GVBisectionPass(3)
    optimal_decomposition_pass = GVOptimalDecompositionPass(3)
    circular_dependency_pass = CircularDependencyBreakerPass()
    greedy_dependency_breaker_pass = GreedyDependencyBreakerPass()
    qubit_dependency_minimizer_pass = QubitDependencyMinimizerPass()
    random_qubit_reuse_pass = RandomQubitReusePass(3)

    result = bisection_pass.run(qernel, 10)
    result = optimal_decomposition_pass.run(qernel, 10)
    result = circular_dependency_pass.run(qernel, 10)
    result = greedy_dependency_breaker_pass.run(qernel, 10)
    #result = qubit_dependency_minimizer_pass.run(qernel, 10)
    result = random_qubit_reuse_pass.run(qernel)

    return result

def test_virtualization(qernel: Qernel) -> List[Qernel]:
    gate_virtualizer = GateVirtualizer()

    results = gate_virtualizer.run(qernel)

    return results

def main():
    #qc = random_circuit(5, 5, max_operands=2, measure=True)
    qc = TwoLocal(5, entanglement='linear', rotation_blocks=["ry"], entanglement_blocks="rzz",reps=1)
    num_params = qc.num_parameters
    qc = qc.bind_parameters(np.random.rand(num_params))
    creg = ClassicalRegister(qc.num_qubits)
    qc.add_register(creg)
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))
    qc = qc.decompose()
    print(qc)

    qernel = Qernel(qc)

    cut_circuit = test_transformation_passes(qernel)
    qernels = test_virtualization(cut_circuit)

    for q in qernels:
        print(q.get_circuit())


main()