import os
from qos.distributed_transpiler.analyser import *
from qos.distributed_transpiler.optimiser import *
from qiskit.circuit.random import random_circuit
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import TwoLocal
from qiskit.providers.fake_provider import *
from qiskit_ibm_provider import IBMProvider
from qiskit import *
from qos.types import Qernel
from qos.virtualizer.virtualizer import GVInstatiator, GVKnitter
from qvm.qvm.virtual_circuit import generate_instantiations
from qos.dag import dag_to_qcg
import matplotlib.pyplot as plt
from supermarq.benchmarks.qaoa_vanilla_proxy import *
from supermarq.benchmarks.ghz import *
from supermarq.converters import *
from FrozenQubits.helper_qaoa import *

def test_analyses_passes(qernel: Qernel) -> None:
    basic_pass = BasicAnalysisPass()
    basic_pass.run(qernel)

    supermarq_feature_pass = SupermarqFeaturesAnalysisPass()
    supermarq_feature_pass.run(qernel)

    dg = DependencyGraphFromDAGPass()
    dg.run(qernel)

    qaoa_analysis = QAOAAnalysisPass()
    qaoa_analysis.run(qernel)


def test_transformation_passes(qernel: Qernel) -> Qernel:
    bisection_pass = GVBisectionPass(3)
    optimal_decomposition_pass = GVOptimalDecompositionPass(3)
    circular_dependency_pass = CircularDependencyBreakerPass()
    greedy_dependency_breaker_pass = GreedyDependencyBreakerPass()
    qubit_dependency_minimizer_pass = QubitDependencyMinimizerPass()
    random_qubit_reuse_pass = RandomQubitReusePass(3)
    optimal_wire_cutting_pass = OptimalWireCuttingPass(4)
    frozen_qubits_pass = FrozenQubitsPass(1)

    result = bisection_pass.run(qernel, 10)
    #result = optimal_decomposition_pass.run(qernel, 10)
    #result = circular_dependency_pass.run(qernel, 10)
    #result = greedy_dependency_breaker_pass.run(qernel, 10)
    #result = qubit_dependency_minimizer_pass.run(qernel, 10)
    #result = random_qubit_reuse_pass.run(qernel)
    #result = optimal_wire_cutting_pass.run(qernel, 10)
    #result = frozen_qubits_pass.run(qernel)

    return result

def test_virtualization(qernel: Qernel) -> List[Qernel]:
    gate_virtualizer = GVInstatiator()

    results = gate_virtualizer.run(qernel)

    return results

def convert_hamiltonian(hamiltonian: List) -> dict:
    new_hamiltonian = {}

    for h in hamiltonian:
        new_hamiltonian[(h[0], h[1])] = h[2]

    return new_hamiltonian

def main2():
    qc_supermarq_bench = GHZ(7)

    qc_supermarq = qc_supermarq_bench.circuit()
    qc_qiskit = cirq_to_qiskit(qc_supermarq)

    qernel = Qernel(qc_qiskit)

    basic_pass = BasicAnalysisPass()
    basic_pass.run(qernel)

    cut_circuit = test_transformation_passes(qernel)
    qernels = test_virtualization(cut_circuit)

    backend = FakePerth()

    to_run = []

    for q in qernels:
        qc_small = q.get_circuit()
        #print(qc_small)
        cqc_small = transpile(qc_small, backend, optimization_level=3)
        to_run.append(cqc_small)

    job = backend.run(to_run, shots=20000)
    qernel.edit_metadata({"shots": 20000})

    results = job.result().get_counts()

    sub_qernel = qernel.get_subqernels()[0]

    for i,sq in enumerate(sub_qernel.get_subqernels()):
        sq.set_results(results[i])

    knitter = GVKnitter()
    print("---------------------------------------")
    print(knitter.run(qernel))


def main3():
    qc_full = QuantumCircuit.from_qasm_file("~/Downloads/FrozenQubits_data_and_sourcecode/experiments/qaoa/sk/gridsearch_100/ideal/3_4_1^P=1.qasm")
    qc_frozen1 = QuantumCircuit.from_qasm_file("~/Downloads/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/sk/gridsearch_100/ideal/3_4_1^M=1_0^P=1.qasm")
    qc_frozen2 = QuantumCircuit.from_qasm_file("~/Downloads/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/sk/gridsearch_100/ideal/3_4_1^M=1_1^P=1.qasm")
    
    qc_full_properties = load_pickle("/home/manosgior/Downloads/FrozenQubits_data_and_sourcecode/experiments/qaoa/sk/gridsearch_100/ideal/3_4_1^P=1.pkl")
    qc_frozen1_properties = load_pickle("/home/manosgior/Downloads/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/sk/gridsearch_100/ideal/3_4_1^M=1_0^P=1.pkl")
    qc_frozen2_properties = load_pickle("/home/manosgior/Downloads/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/sk/gridsearch_100/ideal/3_4_1^M=1_1^P=1.pkl")
    print(qc_full)
    qernel_full = Qernel(qc_full)
    qernel_frozen1 = Qernel(qc_frozen1)
    qernel_frozen2 = Qernel(qc_frozen2)

    qaoa_analysis = QAOAAnalysisPass()

    qaoa_analysis.run(qernel_full)
    qaoa_analysis.run(qernel_frozen1)
    qaoa_analysis.run(qernel_frozen2)

    print("Original circit")
    print(qc_full)
    print("FrozenQubits Hamiltonian:", qc_full_properties["J"])
    print("Our Hamiltonian:", qernel_full.get_metadata()["J"])


    print("Small 1:")
    print(qc_frozen1)
    print("FrozenQubits Hamiltonian:", qc_frozen1_properties["J"])
    print("Our Hamiltonian:", qernel_frozen1.get_metadata()["J"])

    print("Small 2:")
    print(qc_frozen2)
    print("FrozenQubits Hamiltonian:", qc_frozen2_properties["J"])
    print("Our Hamiltonian:", qernel_frozen2.get_metadata()["J"])

def main():
    #qc = random_circuit(5, 5, max_operands=2, measure=True)
    #qc = TwoLocal(5, entanglement='circular', rotation_blocks=["ry"], entanglement_blocks="rzz",reps=1)
    #num_params = qc.num_parameters
    #qc = qc.bind_parameters(np.random.rand(num_params))
    #creg = ClassicalRegister(qc.num_qubits)
    #qc.add_register(creg)
    #qc.measure(range(qc.num_qubits), range(qc.num_qubits))
    #qc = qc.decompose()
    #print(qc)

    #qc_full = QuantumCircuit.from_qasm_file("~/Downloads/FrozenQubits_data_and_sourcecode/experiments/qaoa/sk/gridsearch_100/ideal/3_4_1^P=1.qasm")
    #qc_frozen1 = QuantumCircuit.from_qasm_file("~/Downloads/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/sk/gridsearch_100/ideal/3_4_1^M=1_0^P=1.qasm")
    #qc_frozen2 = QuantumCircuit.from_qasm_file("~/Downloads/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/sk/gridsearch_100/ideal/3_4_1^M=1_1^P=1.qasm")
    #print(qc_full)


    #print(qc_frozen1)
    #print(qc_frozen2)   

    #circuits = [qc_full, qc_frozen1, qc_frozen2]

    provider =  IBMProvider(instance="ibm-q-research-2/tu-munich-1/main")
    backend = provider.get_backend("ibm_perth")
    #backend = FakePerth()

    #ccircuits = transpile(circuits, backend, optimization_level=3)
    #job = backend.run(ccircuits, shots=20000)

    #exit()
   
    qc_supermarq_bench = QAOAVanillaProxy(7)

    qc_supermarq = qc_supermarq_bench.circuit()
    qc_qiskit = cirq_to_qiskit(qc_supermarq)

    to_run = []

    for i in range(5):
        cqc_qiskit = transpile(qc_qiskit, backend, optimization_level=3)
        to_run.append(cqc_qiskit)

    print(qc_qiskit)

    qernel = Qernel(qc_qiskit)

    qaoa_analysis = QAOAAnalysisPass()
    qaoa_analysis.run(qernel)

    cut_circuit = test_transformation_passes(qernel)
    qernels = test_virtualization(cut_circuit)

    for q in qernels:
        qc_small = q.get_circuit()
        #print(qc_small)
        #print(q.get_metadata()["J"])
        for i in range(5):
            cqc_small = transpile(qc_small, backend, optimization_level=3)
            to_run.append(cqc_small)

    job = backend.run(to_run, shots=20000)
    results = job.result().get_counts()

    average_score = 0
    proper_hamiltonian = convert_hamiltonian(qc_supermarq_bench.hamiltonian)

    for i in range(5):
        counts = results[i]
        average_score = average_score + score(qc_qiskit, proper_hamiltonian, counts)
    
    average_score = average_score / 5
    print("big circuit:", average_score)

    average_score_0, average_score_1 = 0, 0

    for i in range(5):
        counts0 = results[i + 5]
        counts1 = results[i + 10]
        average_score_0 = average_score_0 + score(qernels[0].get_circuit(), qernels[0].get_metadata()["J"], counts0)
        average_score_1 = average_score_1 + score(qernels[1].get_circuit(), qernels[1].get_metadata()["J"], counts1)
    
    average_score_0 = average_score_0 / 5
    average_score_1 = average_score_1  / 5
    
    print("small_circuit_0:", average_score_0)
    print("small_circuit_1:", average_score_1)


main3()