import os
from qos.error_mitigator.analyser import *
from qos.error_mitigator.optimiser import *
from qos.error_mitigator.run import *

from qiskit.circuit.random import random_circuit
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import TwoLocal
from qiskit.providers.fake_provider import *
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import *
from qiskit.result import marginal_counts
from qiskit_aer import AerSimulator

import matplotlib.pyplot as plt

from qos.types.types import Qernel
from qos.error_mitigator.virtualizer import GVInstatiator, GVKnitter
from qvm.qvm.virtual_circuit import generate_instantiations
from qvm.examples.fid import calculate_fidelity
from qvm.qvm.run import run_virtual_circuit
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

    return qernel
    #qaoa_analysis = QAOAAnalysisPass()
    #qaoa_analysis.run(qernel)

def test_transformation_passes(qernel: Qernel) -> Qernel:
    bisection_pass = GVBisectionPass(4)
    #optimal_decomposition_pass = GVOptimalDecompositionPass(3)
    #circular_dependency_pass = CircularDependencyBreakerPass()
    #greedy_dependency_breaker_pass = GreedyDependencyBreakerPass()
    #qubit_dependency_minimizer_pass = QubitDependencyMinimizerPass()
    #random_qubit_reuse_pass = RandomQubitReusePass(3)
    #optimal_wire_cutting_pass = OptimalWireCuttingPass(3)
    frozen_qubits_pass = FrozenQubitsPass(1)
    #test_pass = TestPass(3)
    #result = test_pass.run(qernel, 10)

    #result = bisection_pass.run(qernel, 10)
    #result = optimal_decomposition_pass.run(qernel, 10)
    #result = circular_dependency_pass.run(qernel, 10)
    #result = greedy_dependency_breaker_pass.run(qernel, 10)
    #result = qubit_dependency_minimizer_pass.run(qernel, 10)
    #result = random_qubit_reuse_pass.run(qernel)
    #result = optimal_wire_cutting_pass.run(qernel, 10)
    #result = frozen_qubits_pass.run(qernel)
    

    return result

def test_virtualization(qernel: Qernel) -> Qernel:
    gate_virtualizer = GVInstatiator()

    qernel = gate_virtualizer.run(qernel)

    return qernel

def convert_hamiltonian(hamiltonian: List) -> dict:
    new_hamiltonian = {}

    for h in hamiltonian:
        new_hamiltonian[(h[0], h[1])] = h[2]

    return new_hamiltonian

def main2():
    qc_supermarq_bench = GHZ(16)

    qc_supermarq = qc_supermarq_bench.circuit()
    qc_qiskit = cirq_to_qiskit(qc_supermarq)

    qernel = Qernel(qc_qiskit)

    basic_pass = BasicAnalysisPass()
    basic_pass.run(qernel)

    qernel = test_transformation_passes(qernel)
    qernel = test_virtualization(qernel)

    backend = FakeGuadalupe()
    #backend = AerSimulator()
    to_run = []

    tqc_qiskit = transpile(qc_qiskit, backend, optimization_level=3)
    to_run.append(tqc_qiskit)

    sq_list = qernel.get_subqernels()
    for q in sq_list:
        ssqq = q.get_subqernels()
        #print(len(ssqq))
        for q in ssqq:
            qc_small = q.get_circuit()
            cqc_small = transpile(qc_small, backend, optimization_level=3)
            to_run.append(cqc_small)
            print(q.get_circuit())

    job = backend.run(to_run, shots=20000)

    for vsq in qernel.get_virtual_subqernels():
        vsq.edit_metadata({"shots": 20000})
    qernel.edit_metadata({"shots": 20000})

    results = job.result().get_counts()

    sub_qernel = qernel.get_subqernels()[0]

    counter = 1
    for i, sq in enumerate(sub_qernel.get_subqernels()):
        sq.set_results(results[counter])
        counter = counter + 1

    knitter = GVKnitter()
    knitter.run(qernel)

    vsqs = qernel.get_virtual_subqernels() 

    print("Full circuit: ", qc_supermarq_bench.score(results[0]))
    print("Small circuits: ", qc_supermarq_bench.score(vsqs[0].get_results()))

def main3():
    qcs = []
    for i in range(4):
        circ = QuantumCircuit.from_qasm_file("/home/manosgior/Documents/qos/"+str(i)+".qasm") 
        qcs.append(circ)
        print(circ)

    optimal_decomposition_pass = OptimalDecompositionPass(4)
    vqcs = []

    for q in qcs:
        vqcs.append(VirtualCircuit(optimal_decomposition_pass.run(q, 3)))

    for vq in vqcs:
        result, _ = run_virtual_circuit(vq, shots=20000)
        print(calculate_fidelity(vq._circuit, result))

def FrozenQubitsAndQVMExample():
    qc_full = QuantumCircuit.from_qasm_file("/home/manosgior/Downloads/FrozenQubits_data_and_sourcecode/experiments/qaoa/ba/gridsearch_100/ideal/3_7_1^P=1.qasm")

    qc_small = QuantumCircuit.from_qasm_file("/home/manosgior/Downloads/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/ba/gridsearch_100/ideal/3_7_1^M=1_0^P=1.qasm")
    #qc_full_properties = load_pickle("/home/manosgior/Downloads/FrozenQubits_data_and_sourcecode/experiments/qaoa/ba/gridsearch_100/ideal/1_7_1^P=1.pkl")
    print(qc_full)
    #provider = IBMProvider(instance="ibm-q/open/main")
    #backend = provider.get_backend("ibm_brisbane")
    backend = FakeGuadalupe()

    qernel = Qernel(qc_full)
    small_qernel = Qernel(qc_small)

    analyzer = QubitConnectivityGraphFromDAGPass()

    q2 = copy.deepcopy(qernel)
    q2_small = copy.deepcopy(small_qernel)

    analyzer.run(qernel)
    analyzer.run(small_qernel)


    graph = qernel.get_metadata()["qubit_connectivity_graph"]
    graph_small = small_qernel.get_metadata()["qubit_connectivity_graph"]
    
    qr_pass = RandomQubitReusePass(5)
    instantiator = GVInstatiator()
   

    qr_pass.run(q2)
    qr_pass.run(q2_small)

    instantiator.run(q2)
    instantiator.run(q2_small)

    #print(q2_small.get_subqernels()[0].get_circuit())
    

    analyzer.run(q2.get_subqernels()[0])
    analyzer.run(q2_small.get_subqernels()[0])

    

    graph_qr = q2.get_subqernels()[0].get_metadata()["qubit_connectivity_graph"]
    graph_qr_small = q2_small.get_subqernels()[0].get_metadata()["qubit_connectivity_graph"]

    plt.subplot(221)
    nx.draw(graph)
    plt.subplot(222)
    nx.draw(graph_small)
    plt.subplot(223)
    nx.draw(graph_qr)
    plt.subplot(224)
    nx.draw(graph_qr_small)


    plt.savefig("graph.png")
    #plt.savefig("graph_small.png")

    exit()

    dt = DistributedTranspiler(size_to_reach=4, budget=4, methods=["GV", "WC", "QR", "QF"])
    dt.run(qernel)

    print("DT finished")

    qernel = test_virtualization(qernel)

    to_run = []

    cqc_qiskit = transpile(qc_full, backend, optimization_level=3)
    to_run.append(cqc_qiskit)
    #job = backend.run(cqc_qiskit, shots=20000)    

    sqs = qernel.get_subqernels()

    for sq in sqs:
        ssqs = sq.get_subqernels()
        for q in ssqs:
            qc_small = q.get_circuit()
            cqc_small = transpile(qc_small, backend, optimization_level=3)
            to_run.append(cqc_small)

    for vsq in qernel.get_virtual_subqernels():
        vsq.edit_metadata({"shots": 20000})

    print("running")
    job = backend.run(to_run, shots=20000)
    results = job.result().get_counts()

    big_results = results[0]

    print("big circuit: ", score(qernel.get_circuit(), big_results))

    sqs = qernel.get_subqernels()

    counter = 1
    for sq in sqs:
        ssqs = sq.get_subqernels()
        for q in ssqs:
            q.set_results(results[counter])
            counter = counter + 1

    knitting = GVKnitter()
    knitting.run(qernel)          
   
    vsqs = qernel.get_virtual_subqernels()
    sqs = qernel.get_subqernels()

    for i,q in enumerate(sqs):
        print("small_circuit: ", score(q.get_circuit(), vsqs[i].get_results()))

    exit()

def testGVWithGHZ():
    provider =  IBMProvider(instance="ibm-q/open/main")
    backend = provider.get_backend("ibm_brisbane")
    #backend = FakeKolkata()
    qc_supermarq_bench = GHZ(240)

    qc_supermarq = qc_supermarq_bench.circuit()
    qc_qiskit = cirq_to_qiskit(qc_supermarq)
    #qc_qiskit.measure_all(inplace=True)

    qernel = Qernel(qc_qiskit)

    qernel = test_analyses_passes(qernel)
    qernel = test_transformation_passes(qernel)
    qernel = test_virtualization(qernel)


    to_run = []

    #for i in range(5):
     #   cqc_qiskit = transpile(qc_qiskit, backend, optimization_level=3)
      #  to_run.append(cqc_qiskit)

    for sq in qernel.get_subqernels():
        for q in sq.get_subqernels():
            qc_small = q.get_circuit()
            #print(qc_small)
            cqc_small = transpile(qc_small, backend, optimization_level=3)
            to_run.append(cqc_small)

    job = backend.run(to_run, shots=20000)

    #service = QiskitRuntimeService()
    #job = service.job("cma3bjpq34zg008d3ggg")
    results = job.result().get_counts()

    for vsq in qernel.get_virtual_subqernels():
        vsq.edit_metadata({"shots": 20000})

    #average = 0

    ideal_dist = {b * 120: 0.5 for b in ["0", "1"]}

    #for i in range(5):
        #average = average + hellinger_fidelity(ideal_dist, results[i])

    #print("120-q GHZ fidelity:", "{:.3f}".format(average / 5))

    counter = 5
    for sq in qernel.get_subqernels():
        for q in sq.get_subqernels():
            q.set_results(results[counter])
            counter = counter + 1     
    
    knitting = GVKnitter()
    knitting.run(qernel)          

    vsqs = qernel.get_virtual_subqernels() 

    print("4 x 60q GHZ fidelity:", "{:.3f}".format(hellinger_fidelity(ideal_dist, vsqs[0].get_results())))

    exit()

def testFrozenQubits():
    qc_full = QuantumCircuit.from_qasm_file("~/Downloads/FrozenQubits_data_and_sourcecode/experiments/qaoa/ba/gridsearch_100/ideal/3_24_1^P=1.qasm")
    print(qc_full)

    provider =  IBMProvider(instance="ibm-q/open/main")
    backend = provider.get_backend("ibm_brisbane")
    #backend = FakeGuadalupe()

    qernel = Qernel(qc_full)

    qaoa_analysis = QAOAAnalysisPass()
    qaoa_analysis.run(qernel)

    frozen_qubits_pass = FrozenQubitsPass(1)
    qernel = frozen_qubits_pass.run(qernel)

    to_run = []

    cqc_qiskit = transpile(qc_full, backend, optimization_level=3)
    to_run.append(cqc_qiskit)

    for sq in qernel.get_subqernels():
        qc_small = sq.get_circuit()
        print(qc_small)
        cqc_small = transpile(qc_small, backend, optimization_level=3)
        to_run.append(cqc_small)

    job = backend.run(to_run, shots=20000)
    results = job.result().get_counts()

    counts = results[0]

    print("big circuit:", score(qernel.get_circuit(), qernel.get_metadata()["J"], counts))

    counter = 1
    for sq in qernel.get_subqernels():
        sq.set_results(results[counter])
        counter = counter + 1     
    
    vsqs = qernel.get_virtual_subqernels()
    sqs = qernel.get_subqernels()

    print("small_circuit_0:", score(sqs[0].get_circuit(), vsqs[0].get_metadata()["J"], sqs[0].get_results()))
    print("small_circuit_1:", score(sqs[1].get_circuit(), vsqs[1].get_metadata()["J"], sqs[1].get_results()))

    exit()

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

    qc_full = QuantumCircuit.from_qasm_file("~/Downloads/FrozenQubits_data_and_sourcecode/experiments/qaoa/ba/gridsearch_100/ideal/1_7_1^P=1.qasm")

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

    #qc_supermarq_bench = QAOAVanillaProxy(7)

    #qc_supermarq = qc_supermarq_bench.circuit()
    #qc_qiskit = cirq_to_qiskit(qc_supermarq)

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
    
def testDistributedTranspiler():
    dt = DistributedTranspiler(size_to_reach=3, budget=2, methods=["GV", "WC", "QR"])

    #qc_full = QuantumCircuit.from_qasm_file("~/Downloads/FrozenQubits_data_and_sourcecode/experiments/qaoa/ba/gridsearch_100/ideal/1_12_1^P=1.qasm")
    
    qc_supermarq_bench = GHZ(12)
    qc_supermarq = qc_supermarq_bench.circuit()
    qc_full = cirq_to_qiskit(qc_supermarq)

    print(qc_full)
    backend = FakeGuadalupe()

    qernel = Qernel(qc_full)

    dt.run(qernel)

    qernel = test_virtualization(qernel)

    for sqs in qernel.get_subqernels():
        for q in sqs.get_subqernels():
            print(q.get_circuit())

FrozenQubitsAndQVMExample()