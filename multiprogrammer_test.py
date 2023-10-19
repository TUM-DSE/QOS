import os
import random
import sys
from itertools import product

import numpy as np

from benchmarks.circuits import *
from benchmarks.plot.plot import custom_plot_multiprogramming, custom_plot_multiprogramming_relative

from qiskit import *
from qiskit.circuit import QuantumCircuit
from qiskit.providers.fake_provider import *
from qiskit import transpile
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator
from qiskit.quantum_info.analysis import hellinger_fidelity
from qiskit_ibm_provider import IBMProvider

from qos.types import Qernel
from qos.distributed_transpiler.analyser import BasicAnalysisPass, SupermarqFeaturesAnalysisPass
from qos.kernel.multiprogrammer import Multiprogrammer
from qos.kernel.matcher import Matcher

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

def random_selection(circuits: list[QuantumCircuit]):
    circ0 = circuits[random.randint(0, len(circuits) - 1)]
    circ1 = circuits[random.randint(0, len(circuits) - 1)]

    return (circ0, circ1)

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

def transpile_and_prepare(c: QuantumCircuit, reps: int) -> list[QuantumCircuit]:
    toReturn = []

    for i in range(reps):
        toReturn.append(transpile(c, optimization_level=3))

    return toReturn

def execute(circuits: list[QuantumCircuit], backend: BackendV2, shots: int = 20000) -> list[dict[str, int]]:
    job = backend.run(circuits, shots=shots)
    results = job.result()

    return results.get_counts()

def fidelity(counts: dict[str, int], perf_counts) -> float:

    return hellinger_fidelity(counts, perf_counts)

def find_optimal_weights(args: list[str]):
    backend_name = args[3]
    backend = eval(backend_name)()
    lower_limit = int(args[1])
    upper_limit = int(args[2])
    benchmark_circuits = []

    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        for c in circuits:
            benchmark_circuits.append(c)

    possible_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    combinations = list(product(possible_values, repeat=4))
    valid_combinations = [combo for combo in combinations if sum(combo) == 1.0]
    best_weights = []
    best_fid = 0

    for weights in valid_combinations:
        best_circuits = best_selection(benchmark_circuits, weighted=True, weights=weights)
        best_circuits_merged = merge_circs(best_circuits[0], best_circuits[1])

        perfect_results_best = execute(best_circuits_merged, AerSimulator())

        to_execute = transpile_and_prepare(best_circuits_merged, 7)

        counts = execute(to_execute, backend)

        fid = fidelity(counts, perfect_results_best)

        if fid > best_fid:
            best_weights = weights
            best_fid = fid
        
        #print(weights, fid)
    
    print(best_weights, best_fid)

def find_optimal_pair(args: list[str]):
    backend_name = args[3]
    backend = eval(backend_name)()
    lower_limit = int(args[1])
    upper_limit = int(args[2])
    benchmark_circuits = []

    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        for c in circuits:
            benchmark_circuits.append(c)

    best_0 = None
    best_1 = None
    best_fid = 0

    for i in range(len(benchmark_circuits)):
        for j in range(len(benchmark_circuits)):
            merged = merge_circs(benchmark_circuits[i], benchmark_circuits[j])

            perfect_results_best = execute(merged, AerSimulator(), shots=8192)

            to_execute = transpile_and_prepare(merged, 5)

            counts = execute(to_execute, backend, shots=8192)

            fid = fidelity(counts, perfect_results_best)

            if fid > best_fid:                
                best_fid = fid
                best_0 = benchmark_circuits[i]
                best_1 = benchmark_circuits[j]
        
    q0 = Qernel(best_0)
    q1 = Qernel(best_1)

    basic_analysis_pass = BasicAnalysisPass()
    supermarq_analysis_pass = SupermarqFeaturesAnalysisPass()

    basic_analysis_pass.run(q0)
    basic_analysis_pass.run(q1)

    supermarq_analysis_pass(q0)
    supermarq_analysis_pass(q1)

    print(best_fid)
    print(best_0)
    print(best_1)
    print(q0.get_metadata())
    print(q1.get_metadata())


def main(args: list[str]):
    backend_name = args[3]
    backend = eval(backend_name)()
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = int(args[2])
    benchmark_circuits = []
    big_benchmark_circuits = []

    #custom_plot_multiprogramming(["Fidelity vs. Increasing Utilization"], ["Fidelity"], ["Utilization [%]"])
    #custom_plot_multiprogramming_relative(["Fidelity Loss Relative to Baseline"], ["Rel. Fidelity"], ["Utilization [%]"])
    #exit()
    #provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    provider = IBMProvider()

    #backends = provider.backends(min_num_qubits=16, filters=lambda b: b.num_qubits <= 27, simulator=False, operational=True)
    #simulator = provider.get_backend("ibmq_qasm_simulator")
    simulator = AerSimulator()
    #backends = [FakeCairo(), FakeHanoi(), FakeKolkata(), FakeMumbai()]

    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        big_circuits = get_circuits(bench, (lower_limit * 2, lower_limit * 2 + 1))
        
        for c in circuits:
            benchmark_circuits.append(c)
        for bc in big_circuits:
            big_benchmark_circuits.append(bc)

    sum_random_fid = []
    to_execute = []

    for i in range(randomness):    
        random_circuits = random_selection(benchmark_circuits)
        random_circuits_merged = merge_circs(random_circuits[0], random_circuits[1])

        perfect_results_random = execute(random_circuits_merged, simulator)

        sum_random_fid.append(split_counts_bylist(perfect_results_random, [lower_limit, lower_limit]))

        to_execute = to_execute + transpile_and_prepare(random_circuits_merged, 1)
    
    best_circuits = best_selection(benchmark_circuits, weighted=True, weights=[0.25, 0.25, 0.5, 0.0])
    best_circuits_merged = merge_circs(best_circuits[0], best_circuits[1])
    
    perfect_results_best = execute(best_circuits_merged, simulator)
    perfect_results_separate0 = execute(best_circuits[0], simulator)
    perfect_results_separate1 = execute(best_circuits[1], simulator)

    perfect_results_big = [execute(bc, simulator) for bc in big_benchmark_circuits]

    big_circuits_transpiled = [transpile_and_prepare(bc, 1)[0] for bc in big_benchmark_circuits]

    to_execute = to_execute + transpile_and_prepare(best_circuits_merged, 5) + transpile_and_prepare(best_circuits[0], 5) + transpile_and_prepare(best_circuits[1], 5) + big_circuits_transpiled

    counts = execute(to_execute, backend)

    fids_random = []
    for i in range(randomness):
        splitted_counts = split_counts_bylist(counts[i], [lower_limit, lower_limit])
        fid_0 = fidelity(splitted_counts[0], sum_random_fid[i][0])
        fid_1 = fidelity(splitted_counts[1], sum_random_fid[i][1])
        #print(fid_0, fid_1)
        fids_random.append((fid_0 + fid_1) / 2)

    #print(fids_random)
    fid_random = min(fids_random)
    fids_best = []
    for c in counts[randomness:randomness + 5]:
        fids_best.append(split_counts_bylist(c, [lower_limit, lower_limit]))
    
    perf_results_best_splitted = split_counts_bylist(perfect_results_best, [lower_limit, lower_limit])
    fid_best_0 = np.median([fidelity(f[0], perf_results_best_splitted[0]) for f in fids_best])
    fid_best_1 = np.median([fidelity(f[1], perf_results_best_splitted[1]) for f in fids_best])
    #print(fid_best_0, fid_best_1)
    fid_best = (fid_best_0 + fid_best_1) / 2

    fid_separate0 = np.median([fidelity(counts[i], perfect_results_separate0) for i in range(randomness + 5, randomness + 10)])
    fid_separate1 = np.median([fidelity(counts[i], perfect_results_separate1) for i in range(randomness + 10, randomness + 15)])
    #print(fid_separate0, fid_separate1)

    fid_big = np.median([fidelity(counts[i], perfect_results_big[i - (randomness + 15)]) for i in range(randomness + 15, randomness + 15 + len(big_benchmark_circuits))])
    
    print("Random selection fidelity: ", fid_random)
    print("Best selection fidelity: ", fid_best)
    print("Separate fidelity average: ", (fid_separate0 + fid_separate1) / 2)
    print("Big circuits fidelity: ", fid_big)


main(sys.argv)