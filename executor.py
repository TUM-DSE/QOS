from qiskit import QuantumCircuit, transpile, IBMQ, Aer
from qiskit.quantum_info import hellinger_fidelity
import os
import shutil
from datetime import datetime
from qiskit.providers.fake_provider import *
from qiskit.providers.ibmq import AccountProvider
import matplotlib.pyplot as plt
from benchmarks import *
from collections import Counter
import mapomatic as mm
import json
from types import MethodType
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.providers.models import BackendConfiguration
from qiskit_aer.noise import NoiseModel
from benchmarks._utils import _get_ideal_counts
#from ._utils import perfect_counts, fidelity
from qiskit_aer.noise import NoiseModel

# from ._utils import perfect_counts, fidelity

def datetime_to_str(obj):
    """Helper function to convert datetime objects to strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"{type(obj)} not serializable")

def convert_dict_to_json(d, file_path):
    """Recursively convert a dictionary with datetime objects to a JSON file"""
    # Recursively convert nested dictionaries
    for key, value in d.items():
        if isinstance(value, dict):
            convert_dict_to_json(value, file_path)

    # Convert datetime objects to strings using helper function
    d_str = json.dumps(d, default=datetime_to_str, indent=4)

    # Write JSON string to file
    with open(file_path, 'w') as f:
        f.write(d_str)
        
def get_callibration_data(machinename):  
    provider = IBMQ.load_account()
    backends = provider.backends()
    backend = provider.get_backend(machinename)
    
    ranges = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    
    for i,m in enumerate(ranges):

        for j in range(1, m+1):
            t = datetime(day=j, month=i+1, year=2022, hour=10)

            properties = backend.properties(datetime=t)
            
            if properties is None:
                continue
                
            properties = properties.to_dict()

            convert_dict_to_json(properties, "callibration_data/" + machinename + datetime_to_str(t) + ".json")

def move_and_rename_file(src_path, dst_dir, new_name):
    dst_path = os.path.join(dst_dir, new_name)
    # print(dst_path)
    # Move the file to the destination path
    shutil.copy2(src_path, dst_path)


def iterate_files_in_directory():
    provider = IBMQ.load_account()
    benches = [GHZBenchmark(7), VanillaQAOABenchmark(7), FermionicSwapQAOABenchmark(7), HamiltonianSimulationBenchmark(7), BitCodeBenchmark(4), PhaseCodeBenchmark(4), MerminBellBenchmark(7)]
    qcs = [bench.circuit() for bench in benches]
    
    #ranges = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ranges = [31, 31, 30, 31, 30, 31]
    
    backend_names = ["ibm_perth", "ibm_lagos", "ibm_nairobi", "ibm_oslo", "ibmq_jakarta"]
    #backend_names = ["ibm_lagos", "ibm_nairobi", "ibmq_jakarta"]
    city_names = ["perth", "lagos", "nairobi", "oslo", "jakarta"]
    #city_names = ["lagos", "nairobi", "jakarta"]
    fake_backend_names = ["FakePerth", "FakeLagosV2", "FakeNairobiV2", "FakeOslo", "FakeJakartaV2"]
    #fake_backend_names = ["FakeLagos", "FakeNairobi", "FakeJakarta"]
    backends = []
    backend = None
    print(backend_names)
    
    for i,m in enumerate(ranges):
        for j in range(1, m+1):
            t = datetime(day=j, month=i+1, year=2022, hour=10)

            date = datetime_to_str(t)
            fids = []
            
            for k, b in enumerate(backend_names):
                filename =b + date + ".json"
                file_path = os.path.join("callibration_data/" + b, filename)

                if os.path.isfile(file_path):
                    move_and_rename_file(file_path,
                    "/mnt/c/Users/giort/Documents/GitHub/ME/lib/python3.9/site-packages/qiskit/providers/fake_provider/backends/" + city_names[k],
                    "props_" + city_names[k] + ".json")
                    
                #backends.append(eval(fake_backend_names[k])())
                backend = eval(fake_backend_names[k])()
                """
                trans_qcs = [transpile(qc, backends[0], optimization_level=3) for qc in qcs]
                #print(trans_qcs)
                small_qcs = [mm.deflate_circuit(trans_qc) for trans_qc in trans_qcs]
                
                bests = [mm.best_overall_layout(small_qc, backends) for small_qc in small_qcs]
                
                machines = [best[1] for best in bests]
                """
                fids.append([])
                
                for i in range(5):
                    trans_qc = transpile(qcs[4], backend)

                    result = backend.run(trans_qc, shots=8192).result()
                    counts = result.get_counts()
                    fids[k].append(benches[4].score(Counter(counts)))

            for v in fids:
               v.sort()
               
            for i,v in enumerate(fids):
                fids[i] = v[2]
                   
            print(fids)
                
            fids.clear()       
            backends.clear()
    


def execute_on_backends():
    backends = [
        "Casablanca",
        "Jakarta",
        "Lagos",
        "Almaden",
        "Melbourne",
        "Guadalupe",
        "Boeblingen",
        "Singapore",
        "Johannesburg",
        "Cairo",
        "Hanoi",
        "Paris",
        "Sydney",
        "Toronto",
        "Kolkata",
        "Montreal",
        "Cambridge",
        "Washington",
    ]

    bench = HamiltonianSimulationBenchmark(7)
    qc = bench.circuit()

    fids = {}

    for b in backends:
        fids[b] = []
        backend = eval("Fake" + b + "V2()")
        cqc = transpile(qc, backend)

        for i in range(5):
            result = backend.run(cqc, shots=8192).result()
            counts = result.get_counts()
            # avg_fid = avg_fid + hellinger_fidelity(perf_counts, counts)
            fids[b].append(bench.score(Counter(counts)))

        # print(fids)
        # fids.sort()
        print(fids[b])

    means = [np.median(fids[name]) for name in fids]
    stds = [np.std(fids[name]) for name in fids]
    diff_all = np.max(np.abs(np.subtract.outer(means, means)))

    fig, ax = plt.subplots()
    ax.bar(
        fids.keys(),
        means,
        yerr=stds,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
    )
    ax.axhline(np.mean(means), color="green", linewidth=2)
    plt.text(
        0.01,
        0.95,
        "Max difference: {:.2f}".format(diff_all),
        transform=plt.gca().transAxes,
    )
    ax.set_xlabel("IBMQ Backend")
    ax.set_ylabel("Fidelity Score")
    plt.xticks(rotation="vertical")
    ax = plt.gca()
    ax.set_ylim([0.5, 1])
    # ax.set_title('GHZ fidelity score across IBMQ Backends')
    plt.savefig("plot.png", dpi=300, bbox_inches="tight")


def run_all_benchs(dir_path):
    backend = FakeTorontoV2()

    for filename in os.listdir(dir_path):
        print(filename)
        file_path = os.path.join(dir_path, filename)
        qc = QuantumCircuit.from_qasm_file(file_path)
        perf_counts = perfect_counts(qc)
        qc = transpile(qc, backend, optimization_level=3)
        fids = []

        for i in range(5):
            job = backend.run(qc, shots=8192)

            counts = job.result().get_counts()

            fids.append(fidelity(perf_counts, counts))
        print(fids)
        fids.sort()
        print(fids[2])


def execute_benchmarks():
    benchmarks = [
        "HamiltonianSimulation",
        "VQE",
        "VanillaQAOA",
        "GHZ",
        "BitCode",
        "PhaseCode",
        "MerminBell",
        "FermionicSwapQAOA",
    ]

    
    
    provider = IBMQ.load_account()    
    backend = provider.get_backend("ibmq_qasm_simulator")
    
    fake_backend = FakeTorontoV2()
    job_manager = IBMQJobManager()
    
    qbits = [5, 11, 17, 23, 27]
    #qbits = [2, 3]
    
    fids = {}
    circs = []
    


    backend = FakeWashingtonV2()
    backend_noise = NoiseModel.from_backend(backend)

    qbits = [3, 5, 7, 9]
    # qbits = [2, 3]

    fids = {}


    for q in qbits:
        key = str(q) + "q"
        fids[key] = {}

        for b in benchmarks:
            fids[key][b] = []
            print(q, b)
            if b == "PhaseCode" or b == "BitCode":
                bench = eval(b + "Benchmark")(int(q / 2) + 1)
            elif b == "VQE":
                bench = eval(b + "Benchmark")(int(q / 2))
            else:
                bench = eval(b + "Benchmark")(q)

            qc = bench.circuit()

            cqc = transpile(qc, fake_backend, optimization_level=3)
            
            #for i in range(5): 
            qc = bench.circuit()
            cqc = transpile(qc, fake_backend)
            
            job_manager.run(circs, backend, shots=8192)
            
            result = backend.run(cqc, shots=8192).result()
            counts = result.get_counts()
            counts_copy = {}
                           
            
            if isinstance(counts, list):
                fids[key][b].append(bench.score(counts))
            else:
                for (k, v) in counts.items():
                    counts_copy[k.replace(" ", "")] = v
                
                fids[key][b].append(bench.score(Counter(counts_copy)))

            cqc = transpile(qc, backend)

            for i in range(5):
                result = backend.run(cqc, shots=8192).result()
                counts = result.get_counts()
                counts_copy = {}

                if isinstance(counts, list):
                    fids[key][b].append(bench.score(counts))
                else:
                    for (k, v) in counts.items():
                        counts_copy[k.replace(" ", "")] = v

                    fids[key][b].append(bench.score(Counter(counts_copy)))


    # means = { q : [np.median(fids[b]) for b in fids[qbits]] for qbits in fids}
    means = {}
    for q, b in fids.items():
        means[q] = [np.median(fids[q][b]) for b in fids[q]]

    stds = {}
    for q, b in fids.items():
        stds[q] = [np.std(fids[q][b]) for b in fids[q]]
    # diff_all = np.max(np.abs(np.subtract.outer(means, means)))

    group_labels = [
        "Hamiltonian",
        "VQE",
        "VanillaQAOA",
        "GHZ",
        "BitCode",
        "PhaseCode",
        "MerminBell",
        "FermionicQAOA",
    ]

    x = np.arange(len(group_labels))  # the label locations
    width = 0.13  # the width of the bars
    multiplier = 0

    # fig, ax = plt.subplots(layout='constrained')
    fig, ax = plt.subplots()

    for qbits, score in means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=qbits, yerr=stds[qbits])
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_xlabel("Benchmarks")
    ax.set_ylabel("Fidelity Score")
    ax.set_xticks(x + 2 * width, group_labels, rotation=90)
    # Add title and legend
    # ax.set_title("Benchmark score with increasing number of qubits")
    ax.legend()

        
    plt.savefig('plot.png', dpi=300, bbox_inches="tight")
    

def test_qasm_smilator():
    try:
        provider = IBMQ.load_account()
    except e:
        print(e)
        
    backend = provider.get_backend("ibmq_qasm_simulator")
    
    fake_backend = FakeTorontoV2()
    
    
    bench = VanillaQAOABenchmark(5)
    
    qc = bench.circuit()
    
    cqc = transpile(qc, fake_backend, optimization_level=3)
    
    noise_model = NoiseModel.from_backend(fake_backend)
    
    job = backend.run(cqc, shots=8192, noise_model=noise_model)
    
    results = job.result()
    counts = results.get_counts()
    
    #fid = hellinger_fidelity(perf_counts, counts)
    print(bench.score(Counter(counts)))

#test_qasm_smilator()
#execute_benchmarks()

def perfect_counts(original_circuit: QuantumCircuit):
    provider = IBMQ.load_account()
    backend = provider.get_backend("simulator_statevector")
        
    cnt = (
        backend.run(original_circuit, shots=20000).result().get_counts()
    )
    #pdb.set_trace()
    return {k.replace(" ", ""): v for k, v in cnt.items()}
    
def test_counts():
    bench = VanillaQAOABenchmark(12)
    qc = bench.circuit()
    
    counts1 = _get_ideal_counts(qc)
    counts2 = Counter(perfect_counts(qc))
    
    #print(counts1)
    #print(counts2)
    perf_counts = {"0000000000" : 10000, "1111111111": 10000}
    
    fid1 = hellinger_fidelity(perf_counts, counts1)
    fid2 = hellinger_fidelity(perf_counts, counts2)
    
    print(fid1, fid2)


iterate_files_in_directory()

