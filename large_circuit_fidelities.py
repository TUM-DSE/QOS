import csv
import sys
import pandas as pd

from qos.distributed_transpiler.analyser import *

from qiskit.circuit import QuantumCircuit
from qiskit.providers.fake_provider import *
from qiskit import transpile
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator
from qiskit.quantum_info.analysis import hellinger_fidelity
from qiskit_ibm_provider import IBMProvider

from benchmarks.circuits import *
from benchmarks.plot.plot import custom_plot_large_circuit_fidelities

def transpile_and_prepare(c: QuantumCircuit, b: BackendV2, reps: int) -> list[QuantumCircuit]:
    toReturn = []

    for i in range(reps):
        toReturn.append(transpile(c, b, optimization_level=3))

    return toReturn

def execute(circuits: list[QuantumCircuit], backend: BackendV2, shots: int = 20000, save_id = False) -> list[dict[str, int]]:
    job = backend.run(circuits, shots=shots)
    
    job_id = job.job_id()
    print(job_id)

    if save_id:
        with open(job_id, 'w') as file:
            file.write(job_id)

    results = job.result()
    print(results.time_taken)

    return results.get_counts()

def fidelity(counts: dict[str, int], perf_counts) -> float:

    return hellinger_fidelity(counts, perf_counts)

def write_to_csv(filename, data):
    header = ["bench_name", "num_qubits", "depth", "num_nonlocal_gates", "num_measurements", "fidelity", "fidelity_std"]

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(header)
        
        # Write the data
        for row in data:
            writer.writerow(row)

def main(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = int(args[2])
    benchmark_circuits = []

    #provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    #provider = IBMProvider()
    #backend = provider.get_backend("ibmq_kolkata")

    #backend = FakeKolkataV2()
    simulator = provider.get_backend("ibmq_qasm_simulator")
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
    

def main2():
    csv_file_path = "results/large_circuits_solo_"
    dataframes = []

    for i in [4, 8, 12, 16, 20, 24]:
        dataframes.append(pd.read_csv(csv_file_path + str(i) + ".csv"))
    
    custom_plot_large_circuit_fidelities(dataframes, [""], ["Fidelity"], ["Number of Qubits"])

main2()
#main(sys.argv)