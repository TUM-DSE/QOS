from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import hellinger_fidelity
import os
import shutil
from qiskit.providers.fake_provider import *
import matplotlib.pyplot as plt
from benchmarks import *
from collections import Counter
#from ._utils import perfect_counts, fidelity

def move_and_rename_file(src_path, dst_dir, new_name):
    dst_path = os.path.join(dst_dir, new_name)
    #print(dst_path)
    # Move the file to the destination path
    shutil.copy2(src_path, dst_path)



def iterate_files_in_directory(dir_path):    
    bench = VanillaQAOABenchmark(7)
    qc = bench.circuit()

    for filename in os.listdir(dir_path):
        # Construct the full path to the file by joining the directory path and the filename
        file_path = os.path.join(dir_path, filename)
        #print(file_path)
        
        # Check if the path is a file (as opposed to a directory or a symlink)
        if os.path.isfile(file_path):
            move_and_rename_file(file_path, "/mnt/c/Users/giort/Documents/GitHub/ME/lib/python3.9/site-packages/qiskit/test/mock/backends/lagos", "props_lagos.json")

            backend = FakeLagosV2()
            fids = []
            
            for i in range (5):
                qc = transpile(qc, backend)
                
                result = backend.run(qc, shots=8192).result()
                counts = result.get_counts()
                #avg_fid = avg_fid + hellinger_fidelity(perf_counts, counts)
                fids.append(bench.score(Counter(counts)))
                
            fids.sort()
            print(fids[2])
            

#iterate_files_in_directory("/mnt/c/Users/giort/Documents/GitHub/qos/callibration_data")

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
     "Washington"]
     
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
            #avg_fid = avg_fid + hellinger_fidelity(perf_counts, counts)
            fids[b].append(bench.score(Counter(counts)))
                     
        #print(fids)
        #fids.sort()
        print(fids[b])
        
    means = [np.median(fids[name]) for name in fids]
    stds = [np.std(fids[name]) for name in fids]
    diff_all = np.max(np.abs(np.subtract.outer(means, means)))
    

    fig, ax = plt.subplots()
    ax.bar(fids.keys(), means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.axhline(np.mean(means), color='green', linewidth=2)
    plt.text(0.01, 0.95, "Max difference: {:.2f}".format(diff_all),
             transform=plt.gca().transAxes)
    ax.set_xlabel('IBMQ Backend')
    ax.set_ylabel('Fidelity Score')
    plt.xticks(rotation='vertical')
    ax = plt.gca()
    ax.set_ylim([0.5, 1])
    #ax.set_title('GHZ fidelity score across IBMQ Backends')
    plt.savefig('plot.png', dpi=300, bbox_inches="tight")
            


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
        "FermionicSwapQAOA"
    ]
    
    backend = FakeGuadalupeV2()
    
    qbits = [3, 5, 7, 9]
    #qbits = [2, 3]
    
    fids = {}
    
    for q in qbits:
        key = str(q) + "q"
        fids[key] = {}
        
        for b in benchmarks:
            fids[key][b] = []
            print(q, b)
            if b == "PhaseCode" or b == "BitCode":
                bench = eval(b + "Benchmark")(int(q/2) + 1)
            elif b == "VQE":
                bench = eval(b + "Benchmark")(int(q/2))
            else:
                bench = eval(b + "Benchmark")(q)
            
            qc = bench.circuit()
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

                     
    #means = { q : [np.median(fids[b]) for b in fids[qbits]] for qbits in fids}
    means = {}
    for q, b in fids.items():
        means[q] = [np.median(fids[q][b]) for b in fids[q]]
    
    stds = {}
    for q, b in fids.items():
        stds[q] = [np.std(fids[q][b]) for b in fids[q]]
    #diff_all = np.max(np.abs(np.subtract.outer(means, means)))
    
    group_labels = ["Hamiltonian", "VQE", "VanillaQAOA", "GHZ", "BitCode", "PhaseCode", "MerminBell", "FermionicQAOA"]
    
    x = np.arange(len(group_labels))  # the label locations
    width = 0.13  # the width of the bars
    multiplier = 0

    #fig, ax = plt.subplots(layout='constrained')
    fig, ax = plt.subplots()

    for qbits, score in means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=qbits, yerr=stds[qbits])
        #ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_xlabel("Benchmarks")
    ax.set_ylabel("Fidelity Score")
    ax.set_xticks(x + 2 * width, group_labels, rotation=90)
    # Add title and legend
    #ax.set_title("Benchmark score with increasing number of qubits")
    ax.legend()
        
    plt.savefig('plot.png', dpi=300, bbox_inches="tight")
    
execute_benchmarks()