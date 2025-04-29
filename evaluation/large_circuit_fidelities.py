import csv
import sys
import pandas as pd
from time import perf_counter
from multiprocessing import Pool
import copy

from qiskit.circuit import QuantumCircuit
from qiskit.providers.fake_provider import *
from qiskit import transpile
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator
from qiskit.quantum_info.analysis import hellinger_fidelity
from qiskit_ibm_provider import IBMProvider

from benchmarks.circuits import *
from benchmarks.plot.plot import * 

from qos.error_mitigator.analyser import *
from qos.error_mitigator.optimiser import *
from qos.error_mitigator.run import *
from qos.error_mitigator.virtualizer import GVInstatiator, GVKnitter
from qos.multiprogrammer import Multiprogrammer

from qvm.compiler.virtualization.gate_decomp import OptimalDecompositionPass
from qvm.run import run_virtual_circuit
from qvm.quasi_distr import QuasiDistr
import mapomatic.layouts as mply

from benchmarks.circuits.circuits import qaoa
from qos.kernel.estimator import Matcher

def getEffectiveUtilization(qc: QuantumCircuit, backend_num_qubits: int):
    max_duration = 0
    utilization = (qc.num_qubits / backend_num_qubits) * 100

    for i in range(qc.num_qubits):
        if qc.qubit_duration(i) > max_duration:
            max_duration = qc.qubit_duration(i)

    for i in range(qc.num_qubits):
        utilization = utilization - (((qc.qubit_duration(i) / max_duration) * 100) / backend_num_qubits)

    return utilization

def best_selection(circuits: list[QuantumCircuit], weighted: bool = False, weights: list[float] = []):
    basic_analysis_pass = BasicAnalysisPass()
    supermarq_analysis_pass = SupermarqFeaturesAnalysisPass()
    multiprogrammer = Multiprogrammer()
    best_score = 0
    to_return_0 = None
    to_return_1 = None

    for i in range(len(circuits)):
        for j in range(len(circuits)):
            if i != j:
                q0 = Qernel(circuits[i])
                q1 = Qernel(circuits[j])

                basic_analysis_pass.run(q0)
                basic_analysis_pass.run(q1)

                supermarq_analysis_pass.run(q0)
                supermarq_analysis_pass.run(q1)

                score = multiprogrammer.get_matching_score(q0, q1, weighted, weights)

                if score > best_score:
                    to_return_0 = q0.get_circuit()
                    to_return_1 = q1.get_circuit()
                    best_score = score

    return (to_return_0, to_return_1)

def split_counts_bylist(counts, kl):
    counts_list = []
    counts_copy = {}

    for (k, v) in counts.items():
        counts_copy[k.replace(" ", "")] = counts[k]

    for i in range(len(kl)):
        dict = {}

        for (key, value) in counts_copy.items():
            newKey = key[sum(kl) - sum(kl[0 : i + 1]) : sum(kl) - sum(kl[0:i])]
            if newKey in dict:
                continue
            dict.update({newKey: 0})
            for (key2, value2) in counts_copy.items():
                if newKey == key2[sum(kl) - sum(kl[: i + 1]) : sum(kl) - sum(kl[:i])]:
                    dict[newKey] = dict[newKey] + value2

        counts_list.append(dict)

    return counts_list

def merge_circs(c0: QuantumCircuit, c1: QuantumCircuit) -> QuantumCircuit:
    toReturn = QuantumCircuit(
        c0.num_qubits + c1.num_qubits, c0.num_clbits + c1.num_clbits
    )
    qubits1 = [*range(0, c0.num_qubits)]
    clbits1 = [*range(0, c0.num_clbits)]
    qubits2 = [*range(c0.num_qubits, c0.num_qubits + c1.num_qubits)]
    clbits2 = [*range(c0.num_clbits, c0.num_clbits + c1.num_clbits)]

    toReturn.compose(c0, qubits=qubits1, clbits=clbits1, inplace=True)
    toReturn.compose(c1, qubits=qubits2, clbits=clbits2, inplace=True)

    return toReturn

def transpile_and_prepare(c: QuantumCircuit, b: BackendV2, reps: int) -> list[QuantumCircuit]:
    toReturn = []

    for i in range(reps):
        toReturn.append(transpile(c, b, optimization_level=3))

    return toReturn

def execute(circuits: list[QuantumCircuit], backend: BackendV2, shots: int = 20000, save_id = False) -> list[dict[str, int]]:
    job = backend.run(circuits, shots=shots)
    
    job_id = job.job_id()
    

    if save_id:
        print(job_id)
        with open(job_id, 'w') as file:
            file.write(job_id)

    results = job.result()
    print(results.time_taken)

    return results.get_counts()

def fidelity(counts: dict[str, int], perf_counts) -> float:

    return hellinger_fidelity(counts, perf_counts)

def write_to_csv(filename, data):
    #header = ["bench_name", "num_qubits", "depth", "num_nonlocal_gates", "num_measurements", "fidelity", "fidelity_std"]
    header = ["bench_name", "num_qubits", "depth", "num_nonlocal_gates", "num_measurements", "mode_"]

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(header)
        
        # Write the data
        for row in data:
            writer.writerow(row)

def getLargeCircuitFidelities(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = int(args[2])
    benchmark_circuits = []

    provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    #provider = IBMProvider()
    #backend = provider.get_backend("ibmq_kolkata")

    backend = FakeKolkataV2()
    simulator = AerSimulator()
    basic_analysis_pass = BasicAnalysisPass()

    aggr_metadata = []
    perfect_results = []

    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        for c in circuits:
            benchmark_circuits.append(c)
            q = Qernel(c)
            basic_analysis_pass.run(q)
            metadata = q.get_metadata()
            aggr_metadata.append([bench, metadata["num_qubits"], metadata["depth"], metadata["num_nonlocal_gates"], metadata["num_measurements"]])
            perfect_results.append(execute(c, simulator, shots=8192))


    to_execute = []

    for c in benchmark_circuits:
        to_execute = to_execute + transpile_and_prepare(c, backend, randomness)

    results = execute(to_execute, backend, shots=8192, save_id=True)

    counter = 0
    for i in range(len(benchmark_circuits)):
        fids = []
        for j in range(randomness):
            fids.append(fidelity(results[counter], perfect_results[i]))
            counter = counter + 1
        aggr_metadata[i].append(np.median(fids))
        aggr_metadata[i].append(np.std(fids))

    write_to_csv("large_circuits_solo_" + str(lower_limit) + ".csv", aggr_metadata)

def getCuttingResults(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = int(args[2])
    benchmark_circuits = []

    provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    #provider = IBMProvider()
    backend = provider.get_backend("ibm_algiers")

    #backend = FakeKolkataV2()
    simulator = provider.get_backend("ibmq_qasm_simulator")
    #simulator = AerSimulator()
    basic_analysis_pass = BasicAnalysisPass()

    aggr_metadata = []
    perfect_results = []

    for bench in BENCHMARK_CIRCUITS:
        if bench == 'qaoa_r3':
            c = QuantumCircuit.from_qasm_file("/home/manosgior/Downloads/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/k_regular/gridsearch_100/ideal/3_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")
        elif bench == 'qaoa_pl1':    
            c = QuantumCircuit.from_qasm_file("/home/manosgior/Downloads/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/ba/gridsearch_100/ideal/1_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")
        else:
            c = get_circuits(bench, (lower_limit, upper_limit))[0]
               
        benchmark_circuits.append(c)
        q = Qernel(c)
        basic_analysis_pass.run(q)
        metadata = q.get_metadata()
        aggr_metadata.append([bench, metadata["num_qubits"], metadata["depth"], metadata["num_nonlocal_gates"], metadata["num_measurements"]])
        perfect_results.append(execute(c, simulator, shots=8192))

    ready_qernels = []
    dt = DistributedTranspiler(size_to_reach=int(lower_limit/2), budget=3, methods=["GV", "WC", "QR"])
    gate_virtualizer = GVInstatiator()
    knitting = GVKnitter()

    for bc in benchmark_circuits:
        q = Qernel(bc)
        optimized_q = dt.run(q)
        ready_qernels.append(gate_virtualizer.run(optimized_q))
       
    to_execute = []

    for q in ready_qernels:
        sqs = q.get_subqernels()
        
        for sq in sqs:
            ssqs = sq.get_subqernels()
            print(len(ssqs))
            for qq in ssqs:
                qc_small = qq.get_circuit()
                cqc_small = transpile(qc_small, backend, optimization_level=3)
                to_execute.append(cqc_small)

    print(len(to_execute))
    #exit()
    to_execute0 = to_execute[0:150]
    to_execute1 = to_execute[150:]

    for q in ready_qernels:
        for vsq in q.get_virtual_subqernels():
            vsq.edit_metadata({"shots": 8192})

    fids = []
    for i in range(len(ready_qernels)):
        fids.append([])
    
        
    results0 = execute(to_execute0, backend, shots=8192, save_id=True)
    results1 = execute(to_execute1, backend, shots=8192, save_id=True)

    results = results0 + results1

    counter = 0
    for q in ready_qernels:
        for sq in q.get_subqernels():
            ssqs = sq.get_subqernels()
            for qq in ssqs:
                qq.set_results(results[counter])
                counter = counter + 1

        knitting.run(q)

    for i,q in enumerate(ready_qernels):
        vsqs = q.get_virtual_subqernels()
        fids[i].append(fidelity(vsqs[0].get_results(), perfect_results[i]))

    for i in range(len(ready_qernels)):
        aggr_metadata[i].append(np.median(fids[i]))
        aggr_metadata[i].append(np.std(fids[i]))

    write_to_csv("cut_circuits_solo_" + str(lower_limit) + ".csv", aggr_metadata)

def quickTest():
    circuit = QuantumCircuit.from_qasm_file("/home/manosgior/Downloads/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/k_regular/gridsearch_100/ideal/3_24_1^M=2_0^P=1.qasm")
    #circuit =  get_circuits("bv", (24, 25))[0]
    #backend = FakeKolkataV2()
    #provider = IBMProvider()
    provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    simulator = provider.get_backend("ibmq_qasm_simulator")

    job = provider.retrieve_job('cmpd97wz31fg008xynn0')
    noisy_results = job.result().get_counts()

    perfect_results = QuasiDistr.from_counts(execute(circuit, simulator, shots=8192))

    #knitted_results = res_dist.nearest_probability_distribution()

    print("FID: ", fidelity(noisy_results, perfect_results))
    exit()

    decomp_pass = OptimalDecompositionPass(12)

    cut_circ = decomp_pass.run(circuit, 3)
    virt_cut_circ = VirtualCircuit(cut_circ)

    counter = 0
    results = {}
    for frag, frag_circuit in virt_cut_circ.fragment_circuits.items():
        results[frag] = [QuasiDistr.from_counts(noisy_results)]

    with Pool(processes=8) as pool:
        res_dist = virt_cut_circ.knit(results, pool)

def getDTfidelities(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = int(args[2])
    benchmark_circuits = []

    #provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    #provider = IBMProvider()
    #backends = provider.backends(min_num_qubits=16, simulator=False, operational=True)
    #backend = provider.get_backend("ibm_algiers")
    #backend = provider.get_backend("ibmq_kolkata")

    backend = FakeKolkata()
    #simulator = provider.get_backend("ibmq_qasm_simulator")
    simulator = AerSimulator()
    basic_analysis_pass = BasicAnalysisPass()

    aggr_metadata = []
    perfect_results = []

    for bench in BENCHMARK_CIRCUITS:
        if bench == 'qaoa_r3':
            circuits = [QuantumCircuit.from_qasm_file("/home/manosgior/Downloads/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/k_regular/gridsearch_100/ideal/3_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")]
        elif bench == 'qaoa_pl1':
            circuits = [QuantumCircuit.from_qasm_file("/home/manosgior/Downloads/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/ba/gridsearch_100/ideal/1_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")]
        else:
            circuits = get_circuits(bench, (lower_limit, upper_limit))
       
        for c in circuits:
            benchmark_circuits.append(c)
            #print(c)
            #q = Qernel(c)
            #basic_analysis_pass.run(q)
            #metadata = q.get_metadata()
            #aggr_metadata.append([bench, metadata["num_qubits"], metadata["depth"], metadata["num_nonlocal_gates"], metadata["num_measurements"]])
            perfect_results.append(execute(c, simulator, shots=8192))


    ready_qernels = []
    dt = DistributedTranspiler(size_to_reach=8, budget=3, methods=["GV", "WC", "QR", "QF"])
    dt_noQF = DistributedTranspiler(size_to_reach=8, budget=3, methods=["GV", "WC", "QR"])
    gate_virtualizer = GVInstatiator()
    knitting = GVKnitter()

    for i,bc in enumerate(benchmark_circuits):
        q = Qernel(bc)
        #if i == 10:
            #optimized_q = dt_noQF.run(q)
        #else:
            #optimized_q = dt.run(q)
        optimized_q = dt_noQF.run(q)
        ready_qernels.append(gate_virtualizer.run(optimized_q))

    to_execute = []

    for i,q in enumerate(ready_qernels):
        max_depth = 0
        max_cnots = 0
        max_measurements = 0
        max_num_qubits = 0

        sqs = q.get_subqernels()
        print(len(sqs))
        for sq in sqs:
            #if i == 10:
                #print("here")
                #perfect_results[i] = execute(sq.get_circuit(), simulator, shots=8192)
                #print("here")
            ssqs = sq.get_subqernels()
            print(len(ssqs))
            for qq in ssqs:
                qc_small = qq.get_circuit()                
                cqc_small = transpile(qc_small, backend, optimization_level=3, scheduling_method='alap')
                #print(cqc_small)
                to_execute.append(cqc_small)
                #basic_analysis_pass.run(qq)
                #metadata = qq.get_metadata()
                #if metadata["depth"] > max_depth:
                   #max_depth = metadata["depth"]
                #if metadata["num_nonlocal_gates"] > max_cnots:
                    #max_cnots = metadata["num_nonlocal_gates"]
                #if metadata["num_measurements"] > max_measurements:
                    #max_measurements = metadata["num_measurements"]
                #if metadata["num_qubits"] > max_num_qubits:
                    #max_num_qubits = metadata["num_qubits"]
        
        aggr_metadata.append([BENCHMARK_CIRCUITS[i], max_num_qubits, max_depth, max_cnots, max_measurements])

    for q in ready_qernels:
        for vsq in q.get_virtual_subqernels():
            vsq.edit_metadata({"shots": 8192})

    print("Executing")
    print(len(to_execute))

    results = execute(to_execute, backend, shots=8192, save_id=False)

    print("knitting")
    counter = 0
    for q in ready_qernels:
        for sq in q.get_subqernels():
            ssqs = sq.get_subqernels()
            for qq in ssqs:
                qq.set_results(results[counter])
                counter = counter + 1

        knitting.run(q)

    fids = []

    for i,q in enumerate(ready_qernels):
        vsqs = q.get_virtual_subqernels()
        fids.append(fidelity(vsqs[0].get_results(), perfect_results[i]))


    for i in range(len(ready_qernels)):
        aggr_metadata[i].append(fids[i])
        aggr_metadata[i].append(0)

    write_to_csv("cut_circuits_solo_" + str(lower_limit) + ".csv", aggr_metadata)

def getCircuitProperties(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    budget = int(args[2])

    #provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    #provider = IBMProvider()
    #backend = provider.get_backend("ibmq_kolkata")
    backend = FakeKolkataV2()
    half_size = int(args[1])//2
    basic_analysis_pass = BasicAnalysisPass()

    qernels = []
    aggr_metadata = []

    # Original circuit metadata - Qiskit
    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        
        for c in circuits:
            min_cnots = 1000000
            min_depth = 1000000

            for i in range(20):
                tqc = transpile(c, backend, optimization_level=3)
                q = Qernel(tqc)

                basic_analysis_pass.run(q)                    
                metadata = q.get_metadata()

                if metadata["depth"] < min_depth and metadata["num_nonlocal_gates"] < min_cnots:
                    min_depth = metadata["depth"]
                    min_cnots = metadata["num_nonlocal_gates"]
            aggr_metadata.append([bench, circuits[0].num_qubits, min_depth, min_cnots, metadata["num_measurements"], 'qiskit'])

    # Qubit freezing metadata - QF
    # Also storing the qernels after qubit freezing for the next modes
    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        print("qubit freezing - bench: ", bench)
        if bench == 'qaoa_r3':
            small_circ = QuantumCircuit.from_qasm_file("/home/romano/40_Academic/43_QOS/QOS/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/k_regular/gridsearch_100/ideal/3_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")
            min_cnots = 1000000
            min_depth = 1000000

            for i in range(20):
                tqc = transpile(small_circ, backend, optimization_level=3)
                q = Qernel(tqc)
                basic_analysis_pass.run(q)
                metadata = q.get_metadata()
                if metadata["depth"] < min_depth and metadata["num_nonlocal_gates"] < min_cnots:
                    min_depth = metadata["depth"]
                    min_cnots = metadata["num_nonlocal_gates"]
                #if metadata["num_nonlocal_gates"] < min_cnots:
                   # min_cnots = metadata["num_nonlocal_gates"]

            aggr_metadata.append([bench, circuits[0].num_qubits, min_depth, min_cnots, metadata["num_measurements"], 'qf'])
        elif bench == 'qaoa_pl1':
            print("qubit freezing - bench: ", bench)
            small_circ = QuantumCircuit.from_qasm_file("/home/romano/40_Academic/43_QOS/QOS/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/ba/gridsearch_100/ideal/1_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")
            min_cnots = 1000000
            min_depth = 1000000
            for i in range(20):
                tqc = transpile(small_circ, backend, optimization_level=3)
                q = Qernel(tqc)
                basic_analysis_pass.run(q)            
                metadata = q.get_metadata()
                if metadata["depth"] < min_depth and metadata["num_nonlocal_gates"] < min_cnots:
                    min_depth = metadata["depth"]
                    min_cnots = metadata["num_nonlocal_gates"]

            aggr_metadata.append([bench, circuits[0].num_qubits, min_depth, min_cnots, metadata["num_measurements"], 'qf'])
        else:
            print("qubit freezing - bench: ", bench)
            for c in circuits:
                min_cnots = 1000000
                min_depth = 1000000
                for i in range(20):
                    tqc = transpile(c, backend, optimization_level=3)
                    q = Qernel(tqc)
                    
                    basic_analysis_pass.run(q)
                    metadata = q.get_metadata()

                    if metadata["depth"] < min_depth and metadata["num_nonlocal_gates"] < min_cnots:
                        min_depth = metadata["depth"]
                        min_cnots = metadata["num_nonlocal_gates"]
            aggr_metadata.append([bench, circuits[0].num_qubits, min_depth, min_cnots, metadata["num_measurements"], 'FrozenQubits'])

    # Original circuit metadata - Qiskit
    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        
        for c in circuits:
            q = Qernel(c)
            qernels.append(q)

    dt_wc = DistributedTranspiler(size_to_reach=half_size, budget=budget, methods=["WC"])
    gate_virtualizer = GVInstatiator()

    print("cutting - WC")

    #  just gate virtualization
    for i,q in enumerate(qernels):
        #if i == 0 or i == 4:
        try:
            optimized_q = dt_wc.run(q)
        except:
            aggr_metadata.append([BENCHMARK_CIRCUITS[i], circuits[0].num_qubits, aggr_metadata[i][2], aggr_metadata[i][3], aggr_metadata[i][4], 'CutQc'])
            continue
        #else:
            #optimized_q = dt.run(q)
        print("cutting - WC - bench: ", BENCHMARK_CIRCUITS[i])
        final_qernel = gate_virtualizer.run(optimized_q)

        max_depth = 0
        max_cnots = 0

        sqs = final_qernel.get_subqernels()
        print(len(sqs))
        for sq in sqs:
            ssqs = sq.get_subqernels()
            print(len(ssqs))
            for qq in ssqs:
                #if i == 1:
                    #print(qq.get_circuit())
                min_depth = 1000000
                min_cnots = 1000000
                best_score = 100000

                for _ in range(20):
                    tqc = transpile(qq.get_circuit(), backend, optimization_level=3)
                    qqq = Qernel(tqc)
                    basic_analysis_pass.run(qqq)
                    metadata = qqq.get_metadata()

                    score = metadata["depth"] /  aggr_metadata[i][2] + metadata["num_nonlocal_gates"] / aggr_metadata[i][3]
                    if score < best_score:
                        best_score = score
                        min_depth = metadata["depth"]
                        min_cnots = metadata["num_nonlocal_gates"]

                    if min_depth > max_depth:
                        max_depth = min_depth
                    if min_cnots > max_cnots:
                        max_cnots = min_cnots

        aggr_metadata.append([BENCHMARK_CIRCUITS[i], circuits[0].num_qubits, max_depth, max_cnots, metadata["num_measurements"], 'CutQc'])

    print("cutting - QOS")
    
    # Recreating the qernels for the next mode
    qernels = []
    for bench in BENCHMARK_CIRCUITS:
        print("cutting - QOS - bench: ", bench)
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        if bench == 'qaoa_r3':
            small_circ = QuantumCircuit.from_qasm_file("/home/romano/40_Academic/43_QOS/QOS/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/k_regular/gridsearch_100/ideal/3_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")
            qq = Qernel(small_circ)
            qernels.append(qq)
        elif bench == 'qaoa_pl1':
            small_circ = QuantumCircuit.from_qasm_file("/home/romano/40_Academic/43_QOS/QOS/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/ba/gridsearch_100/ideal/1_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")
            qq = Qernel(small_circ)
            qernels.append(qq)
        else:
            qq = Qernel(circuits[0])
            qernels.append(qq)

    dt_noQF = DistributedTranspiler(size_to_reach=half_size, budget=3, methods=["GV", "WC", "QR"])
    gate_virtualizer = GVInstatiator()

    # QOS all - Mode 3
    for i,q in enumerate(qernels):

        optimized_q = dt_noQF.run(q)
        final_qernel = gate_virtualizer.run(optimized_q)

        max_depth = 0
        max_cnots = 0

        sqs = final_qernel.get_subqernels()
        print(len(sqs))
        for sq in sqs:
            ssqs = sq.get_subqernels()
            print(len(ssqs))
            for qq in ssqs:
                min_depth = 1000000
                min_cnots = 1000000
                best_score = 100000

                for j in range(20):
                    tqc = transpile(qq.get_circuit(), backend, optimization_level=3)
                    qqq = Qernel(tqc)
                    basic_analysis_pass.run(qqq)
                    metadata = qqq.get_metadata()

                    score = metadata["depth"] /  aggr_metadata[i][2] + metadata["num_nonlocal_gates"] / aggr_metadata[i][3]
                    if score < best_score:
                        best_score = score
                        min_depth = metadata["depth"]
                        min_cnots = metadata["num_nonlocal_gates"]

                    if min_depth > max_depth:
                        max_depth = min_depth
                    if min_cnots > max_cnots:
                        max_cnots = min_cnots

        aggr_metadata.append([BENCHMARK_CIRCUITS[i], circuits[0].num_qubits, max_depth, max_cnots, metadata["num_measurements"], 'QOS'])
    
    write_to_csv("cut_circuits_reduction" + str(lower_limit) + ".csv", aggr_metadata)

def getCircuitProperties_only_qos(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    budget = int(args[2])
    size_to_reach = int(args[3])

    backend = FakeKolkataV2()
    half_size = int(args[1])//2
    basic_analysis_pass = BasicAnalysisPass()

    qernels = []
    aggr_metadata = []

    # Original circuit metadata - Qiskit
    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        
        for c in circuits:
            min_cnots = 1000000
            min_depth = 1000000

            for i in range(20):
                tqc = transpile(c, backend, optimization_level=3)
                q = Qernel(tqc)

                basic_analysis_pass.run(q)                    
                metadata = q.get_metadata()

                if metadata["depth"] < min_depth and metadata["num_nonlocal_gates"] < min_cnots:
                    min_depth = metadata["depth"]
                    min_cnots = metadata["num_nonlocal_gates"]
            aggr_metadata.append([bench, circuits[0].num_qubits, min_depth, min_cnots, metadata["num_measurements"], 'qiskit'])

    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        if bench == 'qaoa_r3':
            small_circ = QuantumCircuit.from_qasm_file("/home/romano/40_Academic/43_QOS/QOS/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/k_regular/gridsearch_100/ideal/3_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")
            qq = Qernel(small_circ)
            qernels.append(qq)
        elif bench == 'qaoa_pl1':
            small_circ = QuantumCircuit.from_qasm_file("/home/romano/40_Academic/43_QOS/QOS/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/ba/gridsearch_100/ideal/1_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")
            qq = Qernel(small_circ)
            qernels.append(qq)
        else:
            qq = Qernel(circuits[0])
            qernels.append(qq)

    dt_noQF = DistributedTranspiler(size_to_reach=size_to_reach, budget=budget, methods=["GV", "WC", "QR"])
    gate_virtualizer = GVInstatiator()

    for i,q in enumerate(qernels):

        optimized_q = dt_noQF.run(q)
        final_qernel = gate_virtualizer.run(optimized_q)

        max_depth = 0
        max_cnots = 0

        sqs = final_qernel.get_subqernels()
        print(len(sqs))
        for sq in sqs:
            ssqs = sq.get_subqernels()
            print(len(ssqs))
            for qq in ssqs:
                min_depth = 1000000
                min_cnots = 1000000
                best_score = 100000

                for j in range(20):
                    tqc = transpile(qq.get_circuit(), backend, optimization_level=3)
                    qqq = Qernel(tqc)
                    basic_analysis_pass.run(qqq)
                    metadata = qqq.get_metadata()

                    score = metadata["depth"] /  aggr_metadata[i][2] + metadata["num_nonlocal_gates"] / aggr_metadata[i][3]
                    if score < best_score:
                        best_score = score
                        min_depth = metadata["depth"]
                        min_cnots = metadata["num_nonlocal_gates"]

                    if min_depth > max_depth:
                        max_depth = min_depth
                    if min_cnots > max_cnots:
                        max_cnots = min_cnots

        aggr_metadata.append([BENCHMARK_CIRCUITS[i], circuits[0].num_qubits, max_depth, max_cnots, metadata["num_measurements"], 'QOS'])
    
    write_to_csv("qos_budget_eval_" + str(lower_limit) + '_b' + str(budget) + ".csv", aggr_metadata)

def predict_fidelity(circuit, backend, cost_function=None) -> float:
    
    if cost_function is None:
        cost_function = mply.default_cost

    best_out = []

    config = backend.configuration()

    try:
        trans_qc_list = transpile([circuit]*5, backend, optimization_level=3)

        #trans_qc_list = [mm.deflate_circuit(tqc) for tqc in trans_qc_list]
        #trans_qc_list = [tqc for tqc in trans_qc_list if tqc.num_qubits == circuit.num_qubits]
        #print(len(trans_qc_list))

        best_cx_count = [circ.num_nonlocal_gates() for circ in trans_qc_list]
        best_idx = np.argmin(best_cx_count)
        trans_qc = trans_qc_list[best_idx]

    except NameError as e:
        print("[ERROR] - Can't transpile circuit on backend {}".format(backend.name()))
        return 1

    circ = mm.deflate_circuit(trans_qc)

    circ_qubits = circ.num_qubits
    circuit_gates = set(circ.count_ops()).difference({'barrier', 'reset', 'measure'})
    
    #if not circuit_gates.issubset(backend.configuration().basis_gates):
    #    continue
    
    num_qubits = config.num_qubits
    
    layouts = mply.matching_layouts(circ, config.coupling_map, call_limit=int(1e3))
    layout_and_error = mply.evaluate_layouts(circ, layouts, backend,cost_function=cost_function)

    if any(layout_and_error):
        best_error = layout_and_error[0][1]
        for l in layout_and_error:
            if l[1] < best_error:
                best_error = l[1]
    else:
        print("[ERROR] - No layout found for circuit on backend {}".format(backend.name()))
        return 1            

    return 1-best_error

def getCircuitFidelity_only_qos(args: list[str]):

    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    budget = int(args[2])
    size_to_reach = int(args[3])

    backend = FakeKolkataV2()
    backend_alt = FakeKolkata()
    half_size = int(args[1])//2
    basic_analysis_pass = BasicAnalysisPass()

    qernels = []
    aggr_metadata = []

    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        
        for c in circuits:
            min_cnots = 1000000
            min_depth = 1000000

            pred_fid = predict_fidelity(c, backend_alt)
            print(pred_fid)
            tqc = transpile(c, backend, optimization_level=3)
            q = Qernel(tqc)

            aggr_metadata.append([bench, circuits[0].num_qubits, pred_fid, 'qiskit'])

    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        if bench == 'qaoa_r3':
            small_circ = QuantumCircuit.from_qasm_file("/home/romano/40_Academic/43_QOS/QOS/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/k_regular/gridsearch_100/ideal/3_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")
            qq = Qernel(small_circ)
            qernels.append(qq)
        elif bench == 'qaoa_pl1':
            small_circ = QuantumCircuit.from_qasm_file("/home/romano/40_Academic/43_QOS/QOS/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/ba/gridsearch_100/ideal/1_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")
            qq = Qernel(small_circ)
            qernels.append(qq)
        else:
            qq = Qernel(circuits[0])
            qernels.append(qq)

    dt_noQF = DistributedTranspiler(size_to_reach=size_to_reach, budget=budget, methods=["GV", "WC", "QR"])
    gate_virtualizer = GVInstatiator()

    for i,q in enumerate(qernels):

        optimized_q = dt_noQF.run(q)
        final_qernel = gate_virtualizer.run(optimized_q)

        max_depth = 0
        max_cnots = 0
        max_fid = 0

        sqs = final_qernel.get_subqernels()
        print(len(sqs))
        for sq in sqs:
            ssqs = sq.get_subqernels()
            print(len(ssqs))
            for qq in ssqs:
                min_depth = 1000000
                min_cnots = 1000000
                best_score = 100000

                pred_fid = predict_fidelity(qq.get_circuit(), backend_alt)
                
                if pred_fid > max_fid:
                    max_fid = pred_fid

        aggr_metadata.append([BENCHMARK_CIRCUITS[i], circuits[0].num_qubits, max_fid, 'QOS'])
    
    write_to_csv("qos_budget_eval_fid" + str(lower_limit) + '_b' + str(budget) + ".csv", aggr_metadata)

def plot_overheads_avgs_vs_fidelity(args: list[str]):

    #csv_file_path1 = "results/cut_circuits_overheads"
    #csv_file_path2 = "results/qos_budget_eval_fid"
    csv_file_path1 = "fildelity_cost.csv"
    dataframes = []

    dataframes.append(pd.read_csv(csv_file_path1))

    #for i in [12, 24]:
    #    for b in [3, 4, 5]:
    #        dataframes.append(pd.read_csv(csv_file_path1 + str(i) + '_bud' + str(b) + ".csv"))
    #for i in [12, 24]:
    #    for b in [3, 4, 5]:
    #        tmp_df = pd.read_csv(csv_file_path2 + str(i) + '_b' + str(b) + ".csv")
    #        dataframes['bench_name' = tmp_df['bench_name'] & 'budget'] = tmp_df['budget']] = tmp_df['fid']

    #custom_plot_budget_comparison_fidelity(dataframes)
    custom_plot_budget_comparison_fidelity(dataframes)

def generate_qaoa_circuit(num_qubits, connectivity=2):
    # Generate a random graph for Max-Cut problem
    graph = nx.barabasi_albert_graph(num_qubits, connectivity)
    nx.draw(graph)
    plt.savefig("graph")
    q = qaoa(graph)
    dirname = f"benchmarks/circuits/qaoa_pl1"
    with open(f"{dirname}/{q.num_qubits}.qasm", "w") as f:
        f.write(q.qasm())

    return q

def getLargeCircuitCutSizes(args: list[str]):

    '''
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    budget = int(args[3])

    backend = FakeKolkataV2()
    half_size = int(args[2])
    basic_analysis_pass = BasicAnalysisPass()

    pdb.set_trace()
    aggr_metadata = []
    
    bench = get_circuits("ghz", (lower_limit, upper_limit))[0]
    qernel = Qernel(bench)
    
    dt = DistributedTranspiler(size_to_reach=half_size, budget=budget, methods=["WC, QR"])
    gate_virtualizer = GVInstatiator()

    optimized_q = dt.run(qernel)
    final_qernel = gate_virtualizer.run(optimized_q)

    max_frag_size = 0

    sqs = final_qernel.get_subqernels()
    print(len(sqs))

    for sq in sqs:
        ssqs = sq.get_subqernels()
        print(len(ssqs))
        for qq in ssqs:

            #tqc = transpile(qq.get_circuit(), backend, optimization_level=3)
            #qqq = Qernel(tqc)

            #basic_analysis_pass.run(qqq)
            basic_analysis_pass.run(qq)

            #metadata = qqq.get_metadata()
            metadata = qq.get_metadata()

            if max_frag_size < metadata["num_qubits"]:
                maxmax_frag_size_size = metadata["num_qubits"]

    aggr_metadata.append([bench, bench.num_qubits, max_frag_size])

    write_to_csv("large_circuits_frag_sizes" + str(lower_limit) + ".csv", aggr_metadata)
    '''
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    budget = int(args[2])

    #provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    #provider = IBMProvider()
    #backend = provider.get_backend("ibmq_kolkata")
    half_size = int(args[1])//2
    basic_analysis_pass = BasicAnalysisPass()

    qernels = []
    aggr_metadata = []

    # Recreating the qernels for the next mode
    qernels = []
    
    #for bench in BENCHMARK_CIRCUITS:
    #    print("cutting - QOS - bench: ", bench)
    #    circuits = get_circuits(bench, (lower_limit, upper_limit))
    #    qq = Qernel(circuits[0])
    #    qernels.append(qq)

    circuit = generate_qaoa_circuit(lower_limit)
    q = Qernel(circuit)

    techniques = ["QF", "GV"]

    dt = DistributedTranspiler(size_to_reach=half_size, budget=budget, methods=techniques)
    gate_virtualizer = GVInstatiator()
    
    print("cutting")
    optimized_q = dt.run(q)
    print("DT done")

    final_qernel = gate_virtualizer.run(optimized_q)
    print("GV done")

    max_depth = 0
    max_cnots = 0

    sqs = final_qernel.get_subqernels()
    print(len(sqs))

    max_frag_size = 0

    for sq in sqs:
        ssqs = sq.get_subqernels()
        print(len(ssqs))
        for qq in ssqs:
            #tqc = transpile(qq.get_circuit(), backend, optimization_level=3)
            #qqq = Qernel(tqc)
            #qqq = Qernel(qq)
            basic_analysis_pass.run(qq)
            metadata = qq.get_metadata()

            if max_frag_size < metadata["num_qubits"]:
                max_frag_size = metadata["num_qubits"]

    aggr_metadata.append([lower_limit, half_size, max_frag_size, techniques])
    
    write_to_csv("cut_largecircuits" + str(lower_limit) + ".csv", aggr_metadata)

def getCompilationTimes(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = 3
    budget = int(args[2])

    #provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    #provider = IBMProvider()
    #backend = provider.get_backend("ibm_brisbane")

    backend = FakeKolkataV2()

    #basic_analysis_pass = BasicAnalysisPass()

    qernels = []
    aggr_metadata = []
    comp_times = []
    comp_stds = []
    volumes = []
    
    for i,bench in enumerate(BENCHMARK_CIRCUITS):
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        if bench == 'qaoa_r3':
            small_circ = QuantumCircuit.from_qasm_file("/home/romano/40_Academic/43_QOS/QOS/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/k_regular/gridsearch_100/ideal/3_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")
            tmp_comp_times = []
            for i in range(randomness):
                now = perf_counter()
                tqc = transpile(circuits[0], backend, optimization_level=3)
                comp_time = perf_counter() - now
                tmp_comp_times.append(comp_time)

            volumes.append(circuits[0].num_qubits * circuits[0].depth())
            comp_times.append(np.mean(tmp_comp_times))
            comp_stds.append(np.std(tmp_comp_times))
            #q = Qernel(tqc)            
            qq = Qernel(small_circ)
            qernels.append(qq)
        elif bench == 'qaoa_pl1':
            small_circ = QuantumCircuit.from_qasm_file("/home/romano/40_Academic/43_QOS/QOS/FrozenQubits_data_and_sourcecode/experiments/frozenqubits_full/ba/gridsearch_100/ideal/1_" + str(lower_limit) + "_1^M=2_0^P=1.qasm")
            tmp_comp_times = []
            for i in range(randomness):
                now = perf_counter()
                tqc = transpile(circuits[0], backend, optimization_level=3)
                comp_time = perf_counter() - now
                tmp_comp_times.append(comp_time)

            volumes.append(circuits[0].num_qubits * circuits[0].depth())
            comp_times.append(np.mean(tmp_comp_times))
            comp_stds.append(np.std(tmp_comp_times))
            #q = Qernel(tqc)            
            qq = Qernel(small_circ)
            qernels.append(qq)
        else:  
            for c in circuits:
                tmp_comp_times = []
                for i in range(randomness):
                    now = perf_counter()
                    tqc = transpile(c, backend, optimization_level=3)
                    comp_time = perf_counter() - now
                    tmp_comp_times.append(comp_time)
                comp_times.append(np.mean(tmp_comp_times))
                comp_stds.append(np.std(tmp_comp_times))
                #volumes.append(c.num_qubits * c.depth())
                #q = Qernel(tqc)            
                qq = Qernel(c)
                qernels.append(qq)

    #dt = DistributedTranspiler(size_to_reach=12, budget=3, methods=["GV", "WC", "QR", "QF"])
    dt_noQF = DistributedTranspiler(size_to_reach=int(lower_limit/2), budget=budget, methods=["GV", "WC", "QR"])
    gate_virtualizer = GVInstatiator()

    #metadata_delta = []
    new_comp_times = []
    new_comp_times_stds = []
    dt_comp_times = []
    dt_comp_times_stds = []
    #new_volumes = [0 for i in range(len(qernels))]
    #old_volumes = [0 for i in range(len(qernels))]

    for i,q in enumerate(qernels):
        #print("circ: ", i)

        tmp_array = [0]
        qq = copy.deepcopy(q)
        optimized_q = dt_noQF.run(qq)

        dt_comp_times.append(np.mean(tmp_array))
        dt_comp_times_stds.append(np.std(tmp_array))

        final_qernel = gate_virtualizer.run(optimized_q)

        tmp_array = [0 for k in range(randomness)]

        for j in range(randomness):
            sqs = final_qernel.get_subqernels()
            for sq in sqs:
                ssqs = sq.get_subqernels()
                print(len(ssqs))
                #old_volumes[i] = old_volumes[i] + len(ssqs)
                for qq in ssqs:             
                    qc_small = qq.get_circuit()
                    #new_volumes[i] = new_volumes[i] + qc_small.num_qubits * qc_small.depth()
                    now = perf_counter()
                    transpile(qc_small, backend, optimization_level=3)
                    time = perf_counter() - now
                    tmp_array[j] = tmp_array[j] + time
                    #print(tmp_array[j])

        new_comp_times.append(np.mean(tmp_array))
        new_comp_times_stds.append(np.std(tmp_array))
    
        # comp_times = Compilation time
        # comp_stds = Compilation time standard deviation
        # new_comp_times = Compilation time after optimization (sum of all qernels)
        # new_comp_stds = Compilation time standard deviation after optimization
        # dt_comp_times = Compilation time after optimization and qubit freezing
        # dt_comp_times_stds = Compilation time standard deviation after optimization and qubit freezing

        #print(i)
        #print(len(new_comp_times), len(new_comp_times_stds))
        aggr_metadata.append([BENCHMARK_CIRCUITS[i], comp_times[i], comp_stds[i], new_comp_times[i], new_comp_times_stds[i], dt_comp_times[i], dt_comp_times_stds[i]])

    #print(volumes)
    #print(new_volumes)
    #print(old_volumes)
    #df_percentage = [new / old for (new, old) in zip(new_volumes, volumes)]
    #print(df_percentage)
    #print(np.mean(old_volumes))
    #print(np.mean(df_percentage))
    write_to_csv("cut_circuits_overheads" + str(lower_limit) + ".csv", aggr_metadata)

def getMatchingScores(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = int(args[2])

    provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    backends = provider.backends(filters=lambda b: b.num_qubits >= 12, simulator=False, operational=True)
    backends_map = {backend.name : backend for backend in backends}
    print(backends)
    #provider = IBMProvider()
    #backend = provider.get_backend("ibmq_kolkata")
    #backend = FakeKolkataV2()

    benchmark_circuits = []

    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        for c in circuits:
            benchmark_circuits.append(c)
            #q = Qernel(c)
            #basic_analysis_pass.run(q)
            #metadata = q.get_metadata()
            #aggr_metadata.append([bench, metadata["num_qubits"], metadata["depth"], metadata["num_nonlocal_gates"], metadata["num_measurements"]])
            #perfect_results.append(execute(c, simulator, shots=8192))

    
    matcher = Matcher(qpus=backends)

    to_execute = {}

    for b in backends:
        to_execute[b.name] = []

    for bc in benchmark_circuits:
        layout, best_machine, score = matcher.match(bc)[0]
        print(best_machine, layout)
        backend = backends_map[best_machine]
        tqc = transpile(bc, backend, initial_layout=layout)
        to_execute[best_machine].append(tqc)

    print(to_execute)

def getMatchersFidelities(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = int(args[2])

    provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    #backends = [provider.get_backend("ibm_sherbrooke"), provider.get_backend("ibm_brisbane"), provider.get_backend("ibm_auckland")]
    simulator = AerSimulator()
    #backends_map = {backend.name : backend for backend in backends}
    #print(backends)

    benchmark_circuits = []
    perfect_results = []

    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        for c in circuits:
            benchmark_circuits.append(c)
            #q = Qernel(c)
            #basic_analysis_pass.run(q)
            #metadata = q.get_metadata()
            #aggr_metadata.append([bench, metadata["num_qubits"], metadata["depth"], metadata["num_nonlocal_gates"], metadata["num_measurements"]])
            perfect_results.append(execute(c, simulator, shots=8192))

    """
    to_execute_auckland = []

    for i in range(len(benchmark_circuits)):
        to_execute_auckland = to_execute_auckland + transpile_and_prepare(benchmark_circuits[i], backends[2], randomness)

    to_execute_sherbrooke = []

    layouts = [
        [42, 43, 44, 45, 54, 63, 64, 65, 66, 67, 73, 85], [41, 42, 43, 44, 45, 54, 64, 65, 66, 73, 84, 85], [91, 104, 105, 106, 107, 108, 111, 112, 122, 121, 120, 119], [],
        [91, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108], [100, 101, 102, 103, 104, 110, 111, 118, 119, 120, 121, 122],
        [91, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108],[91, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
    ]

    for i in range(len(benchmark_circuits)):
        if i != 3:
            to_execute_sherbrooke = to_execute_sherbrooke + [transpile(benchmark_circuits[i], backends[0], optimization_level=3, initial_layout=layouts[i])] * randomness

    to_execute_brisbane = [transpile(benchmark_circuits[3], backends[1], optimization_level=3, initial_layout=[1, 2, 3, 4, 5, 6, 7, 15, 20, 21, 22, 23])] * randomness

    print(len(to_execute_auckland))
    print(len(to_execute_sherbrooke))
    print(len(to_execute_brisbane))

    job_auck = backends[2].run(to_execute_auckland, shots=8192)
    job_id = job_auck.job_id() 
 
    with open(job_id, 'w') as file:
        file.write(job_id)

    job_sherb = backends[0].run(to_execute_sherbrooke, shots=8192)
    job_id = job_sherb.job_id() 
 
    with open(job_id, 'w') as file:
        file.write(job_id)
    
    job_brisb = backends[1].run(to_execute_brisbane, shots=8192)
    job_id = job_brisb.job_id() 
 
    with open(job_id, 'w') as file:
        file.write(job_id)

    """
    job_auck = provider.retrieve_job('cmtk1sfeskrg008xxvb0')
    job_sherb = provider.retrieve_job('cmtk1szeskrg008xxvbg')
    job_brisb = provider.retrieve_job('cmtk1tq605a0008f1nag')

    results_auck = job_auck.result().get_counts()
    results_sherb = job_sherb.result().get_counts()
    results_brisb = job_brisb.result().get_counts()

    results_agg = results_sherb[0:15] + results_brisb + results_sherb[15:]

    for r in results_brisb:
        results_sherb.insert(3 * randomness, r)

    fidelities_auck = []
    fidelities_auck_std = []
    counter = 0

    #print(len(results_auck))

    for i in range(len(benchmark_circuits)):
        tmp_fids = []
        for j in range(randomness):
            tmp_fids.append(fidelity(perfect_results[i], results_auck[counter]))
            counter = counter + 1
        fidelities_auck.append(np.mean(tmp_fids))
        fidelities_auck_std.append(np.std(tmp_fids))

    fidelities_rest = []
    fidelities_rest_std = []
    counter = 0

    for i in range(len(benchmark_circuits)):
        tmp_fids = []
        for j in range(randomness):
            tmp_fids.append(fidelity(perfect_results[i], results_agg[counter]))
            counter = counter + 1
        fidelities_rest.append(np.mean(tmp_fids))
        fidelities_rest_std.append(np.std(tmp_fids))

    print(fidelities_auck)
    print(fidelities_auck_std)

    print(fidelities_rest)
    print(fidelities_rest_std)

    aggr_metadata = []
    for i in range(len(benchmark_circuits)):
        aggr_metadata.append([BENCHMARK_CIRCUITS[i], fidelities_auck[i], fidelities_auck_std[i], 0 , 0, fidelities_rest[i], fidelities_rest_std[i]])

    write_to_csv("matcher_eval_" + str(lower_limit) + ".csv", aggr_metadata)

    exit()

def getSpatialHeterogeneity(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = int(args[2])

    provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    #backends = provider.backends(filters=lambda b: b.num_qubits == 27, simulator=False, operational=True)
    backends = ["ibm_cairo", "ibm_hanoi", "ibmq_kolkata", "ibm_mumbai", "ibm_algiers", "ibm_auckland"]
    simulator = AerSimulator()
    #backends_map = {backend.name : backend for backend in backends}
    print(backends)

    benchmark = get_circuits("ghz", (lower_limit, upper_limit))[0]

    perfect_results = execute(benchmark, simulator, shots=8192)
    job_ids = ["cmv5ge9fwrrg008851ag", "cmv5gg2jad30008eayb0", "cmv5ggtfwrrg008851b0", "cmv5gkt605a0008f2jf0", "cmv5gmjeskrg008xz26g"]
    jobs = []
    final_job_id = "cmtk1sfeskrg008xxvb0"
    final_job_results = provider.retrieve_job(final_job_id).result().get_counts()

    for jid in job_ids:
        jobs.append(provider.retrieve_job(jid))


    results = []

    for j in jobs:
        results = results + j.result().get_counts()
        #print(j.result().time_taken)

    results = results + final_job_results[5:10]

    """
    for b in backends:
        #tqcs = [transpile(benchmark, b, optimization_level=3)] * randomness
        tqcs = transpile_and_prepare(benchmark, b, randomness)


        job = b.run(tqcs, shots=8192)
        jobs.append(job)
        job_id = job.job_id()
        print(job_id)        
 
        with open(job_id, 'w') as file:
            file.write(b.name + ": " + job_id)
    
    """
    fidelities = []
    fidelities_stds = []
    aggr_metadata = []

    for i in range(0, len(results), 5):
        tmp_fids = []
        for j in range(i, i+5):
            tmp_fids.append(fidelity(results[j], perfect_results))

        fidelities.append(np.mean(tmp_fids))
        fidelities_stds.append(np.std(tmp_fids))
       
    for i in range(len(fidelities)):
        aggr_metadata.append(["ghz", 0, 0, 0 , 0, fidelities[i], fidelities_stds[i]])

    write_to_csv("spatial_hetero_" + str(lower_limit) + ".csv", aggr_metadata)
    print(fidelities)
    print(fidelities_stds)

def getQueueSizes(args: list[str]):
    filename = args[1]

    queue_sizes = {}

    with open(filename, 'r') as file:
        lines = [line for line in file]

    for line in lines:
        data = line.split(sep=";")
        backend, jobs = data[0], int(data[1])

        res = queue_sizes.get(backend)
        if res:
            queue_sizes[backend] = (res[0] + jobs, res[1] + 1)
        else:
            queue_sizes[backend] = (jobs, 1)

    print(queue_sizes)

    for k,v in queue_sizes.items():
        print(k, v[0]/v[1])

    with open("qpu_sizes.csv", 'w') as file:
        file.write("qpu_name, queue_size\n")

        file.write("ibm_lagos," + str(queue_sizes["ibm_lagos"][0] / queue_sizes["ibm_lagos"][1]) + "\n")
        file.write("ibm_nairobi," + str(queue_sizes["ibm_nairobi"][0] / queue_sizes["ibm_nairobi"][1]) + "\n")
        file.write("ibm_perth," + str(queue_sizes["ibm_perth"][0] / queue_sizes["ibm_perth"][1]) + "\n")

        file.write("ibm_algiers," + str(queue_sizes["ibm_algiers"][0] / queue_sizes["ibm_algiers"][1]) + "\n")
        file.write("ibm_auckland," + str(queue_sizes["ibm_auckland"][0] / queue_sizes["ibm_auckland"][1]) + "\n")        
        file.write("ibm_cairo," + str(queue_sizes["ibm_cairo"][0] / queue_sizes["ibm_cairo"][1]) + "\n")
        file.write("ibm_hanoi," + str(queue_sizes["ibm_hanoi"][0] / queue_sizes["ibm_hanoi"][1]) + "\n")
        file.write("ibmq_kolkata," + str(queue_sizes["ibmq_kolkata"][0] / queue_sizes["ibmq_kolkata"][1]) + "\n")
        file.write("ibmq_mumbai," + str(queue_sizes["ibmq_mumbai"][0] / queue_sizes["ibmq_mumbai"][1]) + "\n")

        file.write("ibm_brisbane," + str(queue_sizes["ibm_brisbane"][0] / queue_sizes["ibm_brisbane"][1]) + "\n")
        file.write("ibm_cusco," + str(queue_sizes["ibm_cusco"][0] / queue_sizes["ibm_cusco"][1]) + "\n")
        file.write("ibm_nazca," + str(queue_sizes["ibm_nazca"][0] / queue_sizes["ibm_nazca"][1]) + "\n")
        file.write("ibm_sherbrooke," + str(queue_sizes["ibm_sherbrooke"][0] / queue_sizes["ibm_sherbrooke"][1]) + "\n")

def multiprogrammerEvaluation(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = int(args[2])

    provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    #provider = IBMProvider()
    backend = provider.get_backend("ibmq_kolkata")
    simulator = AerSimulator()
    #simulator =  provider.get_backend("ibmq_qasm_simulator")
    #backend = FakeKolkataV2()
    basic_analysis_pass = BasicAnalysisPass()
    

    benchmark_circuits = []
    aggr_metadata = []
    perfect_results = []
    benchmark_index = {}

    

    for i,bench in enumerate(BENCHMARK_CIRCUITS):
        circuits = get_circuits(bench, (lower_limit, upper_limit))

        for c in circuits:
            c.name = bench
            benchmark_circuits.append(c)
            benchmark_index[bench] = i

            q = Qernel(c)
            basic_analysis_pass.run(q)
            metadata = q.get_metadata()
            aggr_metadata.append([bench, metadata["num_qubits"] * 2, metadata["depth"], metadata["num_nonlocal_gates"], metadata["num_measurements"]])
            perfect_results.append(execute(c, simulator, shots=8192))


    """
    merged_circuits_random = []

    print("---------------------------------")

    benchmark_circuits_copy = copy.deepcopy(benchmark_circuits)

    to_execute = []

    random_index = []

    for i in range(int(len(benchmark_circuits) / 2)):
        circ0 = benchmark_circuits.pop(random.randint(0, len(benchmark_circuits) - 1))
        circ1 = benchmark_circuits.pop(random.randint(0, len(benchmark_circuits) - 1))
        random_index.append((benchmark_index[circ0.name], benchmark_index[circ1.name]))
        
        max_depth = max(circ0.depth(), circ1.depth())
        min_depth = min(circ0.depth(), circ1.depth())
        merged_circuit = merge_circs(circ0, circ1)
        to_execute = to_execute + transpile([merged_circuit] * randomness, backend, optimization_level=3)
        utilization = (circ0.num_qubits / backend.num_qubits) * 100
        effective_utilization = utilization + ((min_depth / max_depth) * 100) * (circ0.num_qubits / backend.num_qubits)
        print("Effective Utilization: ", effective_utilization)

    print("---------------------------------")

    best_index = []

    for i in range(int(len(benchmark_circuits_copy) / 2)):
        (circ0, circ1) = best_selection(benchmark_circuits_copy, True, [0.25, 0.25, 0.5, 0.0])
        #(circ0, circ1) = best_selection(benchmark_circuits_copy, True, [1.0, 0.0, 0.0, 0.0])
        benchmark_circuits_copy.remove(circ0)
        benchmark_circuits_copy.remove(circ1)

        best_index.append((benchmark_index[circ0.name], benchmark_index[circ1.name]))

        max_depth = max(circ0.depth(), circ1.depth())
        min_depth = min(circ0.depth(), circ1.depth())         
        merged_circuit = merge_circs(circ0, circ1)
        to_execute = to_execute + transpile([merged_circuit] * randomness, backend, optimization_level=3)
        utilization = (circ0.num_qubits / backend.num_qubits) * 100
        effective_utilization = utilization + ((min_depth / max_depth) * 100) * (circ0.num_qubits / backend.num_qubits)
        print("Effective Utilization: ", effective_utilization)

    print(random_index)
    print(best_index)

    results = execute(to_execute, backend, 8192, True)

    """
    job = provider.retrieve_job('cmrkwq2b37pg008q8ppg')
    results = job.result().get_counts()
    random_index = [(1, 9), (7, 2), (4, 5), (6, 8), (3, 0)]
    best_index = [(3, 6), (0, 5), (1, 8), (7, 9), (2, 4)]

    counter = 0
    average_random_fidelities = [0 for i in range(len(perfect_results))]
    std_random_fidelities = [0 for i in range(len(perfect_results))]
    average_best_fidelities = [0 for i in range(len(perfect_results))]
    std_best_fidelities = [0 for i in range(len(perfect_results))]

    for i in range(len(random_index)):
        tmp_fids0 = []
        tmp_fids1 = []
        for j in range(randomness):
            splitted_counts = split_counts_bylist(results[counter], [lower_limit, lower_limit])
            tmp_fids0.append(fidelity(splitted_counts[0], perfect_results[random_index[i][0]]))
            tmp_fids1.append(fidelity(splitted_counts[1], perfect_results[random_index[i][1]]))
            counter = counter + 1
        
        average_random_fidelities[random_index[i][0]] = np.mean(tmp_fids0)
        average_random_fidelities[random_index[i][1]] = np.mean(tmp_fids1)
        std_random_fidelities[random_index[i][0]] = np.std(tmp_fids0)
        std_random_fidelities[random_index[i][1]] = np.std(tmp_fids0)

    for i in range(len(best_index)):
        tmp_fids0 = []
        tmp_fids1 = []
        for j in range(randomness):
            splitted_counts = split_counts_bylist(results[counter], [lower_limit, lower_limit])
            tmp_fids0.append(fidelity(splitted_counts[0], perfect_results[best_index[i][0]]))
            tmp_fids1.append(fidelity(splitted_counts[1], perfect_results[best_index[i][1]]))
            counter = counter + 1
        
        average_best_fidelities[best_index[i][0]] = np.mean(tmp_fids0)
        average_best_fidelities[best_index[i][1]] = np.mean(tmp_fids1)
        std_best_fidelities[best_index[i][0]] = np.std(tmp_fids0)
        std_best_fidelities[best_index[i][1]] = np.std(tmp_fids0)

    for i in range(len(average_best_fidelities)):
        aggr_metadata[i].append(average_random_fidelities[i])
        aggr_metadata[i].append(average_best_fidelities[i])

    write_to_csv("multiprogrammer_test" + str(lower_limit * 2) + ".csv", aggr_metadata)
    #print(np.mean(average_random_fidelities), np.mean(average_best_fidelities))

def end_to_end_eval(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = int(args[2])
    benchmark_circuits = []

    #provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    #provider = IBMProvider()
    #backends = provider.backends(min_num_qubits=16, simulator=False, operational=True)
    #backend = provider.get_backend("ibmq_kolkata")

    backend = FakeKolkataV2()
    #simulator = provider.get_backend("ibmq_qasm_simulator")
    simulator = AerSimulator()
    basic_analysis_pass = BasicAnalysisPass()

    aggr_metadata = []
    perfect_results = []

    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        for c in circuits:
            benchmark_circuits.append(c)
            q = Qernel(c)
            basic_analysis_pass.run(q)
            metadata = q.get_metadata()
            aggr_metadata.append([bench, metadata["num_qubits"], metadata["depth"], metadata["num_nonlocal_gates"], metadata["num_measurements"]])
            perfect_results.append(execute(c, simulator, shots=8192))

    ready_qernels = []
    dt = DistributedTranspiler(size_to_reach=6, budget=3, methods=["GV", "WC", "QR", "QF"])
    gate_virtualizer = GVInstatiator()
    #matcher = Matcher(qpus=backends)
    knitting = GVKnitter()

    for bc in benchmark_circuits:
        q = Qernel(bc)
        optimized_q = dt.run(q)
        ready_qernels.append(gate_virtualizer.run(optimized_q))
       

    to_execute = []
    to_unpack = []
    counter = 0
    tmp_circuit = None

    print("merging")
    for i,q in enumerate(ready_qernels):
        sqs = q.get_subqernels()
        for sq in sqs:
            if i == 0 or i == 4:
                perfect_results[i] = execute(sq.get_circuit(), simulator, shots=8192)
            ssqs = sq.get_subqernels()
            for qq in ssqs:
                qc_small = qq.get_circuit()
                if tmp_circuit == None:
                    tmp_circuit = qc_small
                    to_unpack.append([])
                    to_unpack[counter].append(tmp_circuit.num_qubits)
                else:
                    if tmp_circuit.num_qubits + qc_small.num_qubits <= backend.num_qubits:
                        tmp_circuit = merge_circs(tmp_circuit, qc_small)
                        to_unpack[counter].append(qc_small.num_qubits)
                    else:
                        to_execute.append(tmp_circuit)
                        tmp_circuit = qc_small
                        to_unpack.append([])
                        counter = counter + 1
                        to_unpack[counter].append(tmp_circuit.num_qubits)
                        
                #cqc_small = transpile(qc_small, backend, optimization_level=3)
                #to_execute.append(cqc_small)

    if (len(to_execute) < len(to_unpack)):
        to_execute.append(tmp_circuit)

    print("transpiling")
    to_execute_copy = []
    for c in to_execute:
        to_execute_copy.append(transpile(c, backend, optimization_level=3))
    
    for q in ready_qernels:
        for vsq in q.get_virtual_subqernels():
            vsq.edit_metadata({"shots": 8192})

    fids = []
    for i in range(len(ready_qernels)):
        fids.append([])
       
    print("executing")
    results = execute(to_execute, backend, shots=8192, save_id=False)

    unpacked_results = []

    print("unpacking")
    for i,r in enumerate(results):
        lists = split_counts_bylist(r, to_unpack[i])
        for l in lists:
            unpacked_results.append(l)


    print("knitting")
    counter = 0
    for q in ready_qernels:
        for sq in q.get_subqernels():
            ssqs = sq.get_subqernels()
            for qq in ssqs:
                qq.set_results(unpacked_results[counter])
                counter = counter + 1

        knitting.run(q)

    for i,q in enumerate(ready_qernels):
        vsqs = q.get_virtual_subqernels()
        if i == 0 or i == 4:
            fids.append(fidelity(vsqs[3].get_results(), perfect_results[i]))
        else:
            fids.append(fidelity(vsqs[0].get_results(), perfect_results[i]))

    for i in range(len(ready_qernels)):
        aggr_metadata[i].append(np.median(fids[i]))
        aggr_metadata[i].append(np.std(fids[i]))

    write_to_csv("cut_circuits_MP_" + str(lower_limit) + ".csv", aggr_metadata)

def plot_large_circuits_solo():
    csv_file_path = "results/large_circuits_solo_"
    dataframes = []

    for i in [4, 8, 12, 16, 20, 24]:
        dataframes.append(pd.read_csv(csv_file_path + str(i) + ".csv"))
    
    custom_plot_large_circuit_fidelities(dataframes, [""], ["Fidelity"], ["Number of Qubits"])

def plot_relative_12q_solo():
    dataframes = [pd.read_csv("results/cut_circuits_solo_12.csv"), pd.read_csv("results/large_circuits_solo_12.csv")]

    custom_plot_small_circuit_relative_fidelities(dataframes, [""], ["Relative Fidelity [x]"], ["Benchmark"])

def plot_relative_properties():
    csv_file_path = "./cut_circuits_reduction"
    dataframes = []

    for i in [12, 24]:
        dataframes.append(pd.read_csv(csv_file_path + str(i) + ".csv"))

    #custom_plot_small_circuit_relative_properties(dataframes, ["(a) Circuit Depth", "(b) CNOTs"], ["Rel. Depth", "Rel. Number of CNOTs"], ["Number of Qubits", "Number of Qubits"])
#    custom_custom_plot_small_circuit_relative_properties(dataframes, ["(a) Circuit Depth", "(b) CNOTs"], ["Rel. Depth", "Rel. Number of CNOTs"], ["Number of Qubits", "Number of Qubits"])

    custom_custom_plot_small_circuit_relative_properties_barplot(dataframes, ["(a) Circuit Depth", "(b) CNOTs"], ["Rel. Depth", "Rel. Number of CNOTs"], ["Number of Qubits", "Number of Qubits"])

def plot_budget_comparison():
    csv_file_path = "./qos_budget_eval_"
    dataframes = []
    circuit_size = 12
    budgets = [3, 4, 5]

    for i in budgets:
        dataframes.append(pd.read_csv(csv_file_path + str(circuit_size) + '_b' + str(i) + ".csv"))

    custom_plot_budget_comparison(dataframes, [""], ["Fidelity"], ["Budget"])

def plot_budget_vs_fidelity():
    csv_file_path = "./qos_budget_eval_fid"
    dataframes = []
    circuit_size = [12, 24]
    budgets = [3, 4, 5]

    for i in budgets:
        for j in circuit_size:
            dataframes.append(pd.read_csv(csv_file_path + str(j) + '_b' + str(i) + ".csv"))

    plot_budget_comparison_fidelity(dataframes, [""], ["Fidelity"], ["Budget"])

def plot_overheads():
    csv_file_path = "results/cut_circuits_overheads"
    dataframes = []

    for i in [12, 24]:
        dataframes.append(pd.read_csv(csv_file_path + str(i) + ".csv"))

    custom_plot_small_circuit_overheads(dataframes, [""], ["Runtime [s]", "Number of Circuits"], ["Number of Qubits", "Number of Qubits"])

def plot_cut_fidelities():
    dataframes = []

    small_file_path = "results/cut_circuits_solo_"
    large_file_path = "results/large_circuits_solo_"

    for i in [12, 24]:
        dataframes.append(pd.read_csv(large_file_path + str(i) + ".csv"))
        dataframes.append(pd.read_csv(small_file_path + str(i) + ".csv"))

    custom_plot_small_circuit_fidelities(dataframes, ["QAOA-R3", "BV", "GHZ", "HS-1", "QAOA-P1", "QSVM", "TL-1", "VQE-1", "W-STATE"], ["Fidelity"], ["Number of Qubits"])

def plot_multiprogrammer():
    dataframes = []

    mc_file_path = "results/multiprogrammer_test"
    baseline_file_path = "results/large_circuits_solo_"

    for i in [8, 16, 24]:
        dataframes.append(pd.read_csv(mc_file_path + str(i) + ".csv"))
        dataframes.append(pd.read_csv(baseline_file_path + str(i) + ".csv"))

    custom_plot_multiprogrammer(dataframes, [], ["Fidelity"], ["Utilization [%]"])

def plot_multiprogrammer_relative():
    dataframes = []

    mc_file_path = "results/multiprogrammer_test"
    baseline_file_path = "results/large_circuits_solo_"

    for i in [8, 16, 24]:
        dataframes.append(pd.read_csv(mc_file_path + str(i) + ".csv"))
        dataframes.append(pd.read_csv(baseline_file_path + str(i) + ".csv"))

    for i in [4, 8, 12]:
        dataframes.append(pd.read_csv(mc_file_path + str(i * 2) + ".csv"))
        dataframes.append(pd.read_csv(baseline_file_path + str(i) + ".csv"))    

    custom_plot_multiprogrammer_relative(dataframes, ["(a) Impact on Fidelity", "(b) Effective Utilization", "(c) Relative Fidelity"], ["Fidelity", "Effective Utilization [%]", "Rel. Fidelity"], ["Utilization [%]", "Ideal Utilization [%]", "Utilization [%]"])

def plot_utilization_problem():
    custom_plot_baseline_utilizations([], [""], ["Utilization [%]"], ["Benchmark"])

def plot_matcher_performance():
    filename = "matcher_eval_12.csv"

    dataframe = pd.read_csv(filename)

    custom_plot_matcher([dataframe], [""], ["Fidelity"], ["Benchmark"])

def plot_spatial_hetero():
    filename = "results/spatial_hetero_12.csv"

    dataframe = pd.read_csv(filename)

    custom_plot_spatial_hetero([dataframe], [""], ["Fidelity"], ["QPU"])

def plot_scal_spatial_hetero():
    csv_file_path = "results/large_circuits_solo_"
    dataframes = []

    for i in [4, 8, 12, 16, 20, 24]:
        dataframes.append(pd.read_csv(csv_file_path + str(i) + ".csv")) 

    dataframes.append(pd.read_csv("results/spatial_hetero_12.csv"))

    custom_plot_scal_spatial_hetero(dataframes, ["(a) Scalability", "(b) Spatial Heterogeneity"], ["Fidelity"], ["Number of Qubits", "IBM QPU"])

def plot_temp_util_load():
    dataframes = []

    dataframes.append(pd.read_csv("results/perth_fids_6.csv"))
    dataframes.append(pd.read_csv("results/qpu_queue_sizes.csv"))

    custom_plot_temp_util_load(dataframes, ["(a) Temporal Variance", "(b) Utilization", "(c) QPU Load"], ["Fidelity", "Utilization [%]", "Number of Pending Jobs"], ["Calibration Day", "Benchmark", "IBM QPU"])

def plot_dt_util_estim():
    csv_file_path = "results/cut_circuits_overheads"
    mc_file_path = "results/multiprogrammer_test"
    baseline_file_path = "results/large_circuits_solo_"
    filename = "results/matcher_eval_12.csv"
    dataframes = []

    for i in [12, 24, 40]:
        dataframes.append(pd.read_csv(csv_file_path + str(i) + ".csv"))

    for i in [8, 16, 24]:
        dataframes.append(pd.read_csv(mc_file_path + str(i) + ".csv"))
        dataframes.append(pd.read_csv(baseline_file_path + str(i) + ".csv"))

    dataframes.append(pd.read_csv(filename))

    custom_plot_dt_mp_estim(dataframes, ["(a) Distributed Transpiler Overheads", "(b) QOS Multi-progammer", "(c) QOS Estimator"], ["Runtime [s]", "Fidelity", "Fidelity"], ["Number of Qubits", "Utilization [%]", "Benchmark"])

def plot_estimator():
    filename = "results/matcher_eval_12.csv"
    dataframes = []
    
    dataframes.append(pd.read_csv(filename))
    
    custom_plot_estimator(dataframes, [], ["Fidelity"], ["Benchmark"])

#qosDTeval(sys.argv)
#plot_cut_fidelities()
#quickTest()
#getDTfidelities(sys.argv)
#multiprogrammerEvaluation(sys.argv)
#getMatchersFidelities(sys.argv)
#plot_matcher_performance()
#getSpatialHeterogeneity(sys.argv)
#plot_utilization_problem()
#plot_spatial_hetero()
#getSpatialHeterogeneity(sys.argv)#
#plot_multiprogrammer()
#plot_multiprogrammer_relative()
#plot_overheads()
#print(sys.argv)
#pdb.set_trace()
#getCircuitFidelity_only_qos(sys.argv)
#getCircuitProperties(sys.argv)
#getCircuitProperties_only_qos(sys.argv)
plot_relative_properties()
#plot_budget_vs_fidelity()
#plot_budget_comparison()
#getLargeCircuitCutSizes(sys.argv)
#plot_scal_spatial_hetero()
#getCuttingResults(sys.argv)
#getQueueSizes(sys.argv)
#plot_temp_util_load()
#getCircuitProperties(sys.argv)
#plot_cut_fidelities()
#getCompilationTimes(sys.argv)
#plot_overheads_avgs_vs_fidelity(sys.argv)
#plot_dt_util_estim()
#plot_estimator()
