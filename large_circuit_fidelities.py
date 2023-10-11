import csv
import sys
import pandas as pd

from qiskit.circuit import QuantumCircuit
from qiskit.providers.fake_provider import *
from qiskit import transpile
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator
from qiskit.quantum_info.analysis import hellinger_fidelity
from qiskit_ibm_provider import IBMProvider

from benchmarks.circuits import *
from benchmarks.plot.plot import custom_plot_large_circuit_fidelities, custom_plot_small_circuit_relative_fidelities, custom_plot_small_circuit_relative_properties

from qos.distributed_transpiler.analyser import *
from qos.distributed_transpiler.optimiser import *
from qos.distributed_transpiler.run import *
from qos.distributed_transpiler.virtualizer import GVInstatiator, GVKnitter

#from qos.kernel.matcher import Matcher

def transpile_and_prepare(c: QuantumCircuit, b: BackendV2, reps: int) -> list[QuantumCircuit]:
    toReturn = []

    for i in range(reps):
        toReturn.append(transpile(c, b, optimization_level=3))

    return toReturn

def execute(circuits: list[QuantumCircuit], backend: BackendV2, shots: int = 20000, save_id = False) -> list[dict[str, int]]:
    job = backend.run(circuits, shots=shots)
    
    job_id = job.job_id()
    #print(job_id)

    if save_id:
        with open(job_id, 'w') as file:
            file.write(job_id)

    results = job.result()
    #print(results.time_taken)

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
    

def getCuttingResults(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = int(args[2])
    benchmark_circuits = []

    #provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    provider = IBMProvider()
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
            for qq in ssqs:
                qc_small = qq.get_circuit()
                cqc_small = transpile(qc_small, backend, optimization_level=3)
                to_execute.append(cqc_small)


    for q in ready_qernels:
        for vsq in q.get_virtual_subqernels():
            vsq.edit_metadata({"shots": 8192})

    fids = []
    for i in range(len(ready_qernels)):
        fids.append([])
    
    for i in range(randomness):    
        results = execute(to_execute, backend, shots=8192, save_id=False)

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

def getCircuitProperties(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = int(args[2])

    #provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    #provider = IBMProvider()
    #backend = provider.get_backend("ibmq_kolkata")
    backend = FakeKolkataV2()

    basic_analysis_pass = BasicAnalysisPass()

    qernels = []
    aggr_metadata = []
    
    for bench in BENCHMARK_CIRCUITS:
        circuits = get_circuits(bench, (lower_limit, upper_limit))
        for c in circuits:
            tqc = transpile(c, backend, optimization_level=3)
            q = Qernel(tqc)
            qq = Qernel(c)
            basic_analysis_pass.run(q)
            qernels.append(qq)
            metadata = q.get_metadata()
            aggr_metadata.append([bench, metadata["num_qubits"], metadata["depth"], metadata["num_nonlocal_gates"], metadata["num_measurements"], 0, 0])

    dt = DistributedTranspiler(size_to_reach=12, budget=3, methods=["GV", "WC", "QR", "QF"])
    gate_virtualizer = GVInstatiator()

    #metadata_delta = []

    for i,q in enumerate(qernels):
        print(i)
        optimized_q = dt.run(q)
        final_qernel = gate_virtualizer.run(optimized_q)

        max_depth = 0
        max_cnots = 0

        sqs = final_qernel.get_subqernels()

        for sq in sqs:
            ssqs = sq.get_subqernels()
            for qq in ssqs:
                tqc = transpile(qq.get_circuit(), backend, optimization_level=3)
                qqq = Qernel(tqc)
                basic_analysis_pass.run(qqq)
                metadata = qqq.get_metadata()
                if metadata["depth"] > max_depth:
                    max_depth = metadata["depth"]

                if metadata["num_nonlocal_gates"] > max_cnots:
                    max_cnots = metadata["num_nonlocal_gates"]

        aggr_metadata[i][2] = max_depth / aggr_metadata[i][2]
        aggr_metadata[i][3] = max_cnots / aggr_metadata[i][3]

    write_to_csv("cut_circuits_reduction" + str(lower_limit) + ".csv", aggr_metadata)

def end_to_end_eval(args: list[str]):
    lower_limit = int(args[1])
    upper_limit = lower_limit + 1
    randomness = int(args[2])
    benchmark_circuits = []

    provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    #provider = IBMProvider()
    backends = provider.backends(simulator=False, operational=True)
    #backend = provider.get_backend("ibmq_kolkata")

    backend = FakeGuadalupeV2()
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
    dt = DistributedTranspiler(size_to_reach=4, budget=4, methods=["GV", "WC", "QR", "QF"])
    gate_virtualizer = GVInstatiator()
    matcher = Matcher(qpus=backends)
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
            for qq in ssqs:
                qc_small = qq.get_circuit()
                cqc_small = transpile(qc_small, backend, optimization_level=3)
                to_execute.append(cqc_small)

    print(len(to_execute))
    exit()

    for q in ready_qernels:
        for vsq in q.get_virtual_subqernels():
            vsq.edit_metadata({"shots": 8192})

    fids = []
    for i in range(len(ready_qernels)):
        fids.append([])
    
    for i in range(randomness):    
        results = execute(to_execute, backend, shots=8192, save_id=False)

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
    csv_file_path = "results/cut_circuits_reduction"
    dataframes = []

    for i in [8, 16, 24]:
        dataframes.append(pd.read_csv(csv_file_path + str(i) + ".csv"))

    custom_plot_small_circuit_relative_properties(dataframes, ["CNOTs", "Circuit Depth"], ["Rel. Depth", "Rel. Number of CNOTs"], ["Number of Qubits"])


end_to_end_eval(sys.argv)
