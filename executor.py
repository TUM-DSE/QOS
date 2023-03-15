from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import hellinger_fidelity
import os
import shutil
from qiskit.providers.fake_provider import *
from benchmarks import *
from collections import Counter

def move_and_rename_file(src_path, dst_dir, new_name):
    dst_path = os.path.join(dst_dir, new_name)
    #print(dst_path)
    # Move the file to the destination path
    shutil.copy2(src_path, dst_path)



def iterate_files_in_directory(dir_path):    
    bench = GHZBenchmark(7)
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
     
    bench = VanillaQAOABenchmark(7)
    qc = bench.circuit()
    
    for b in backends:
        fids = []
        backend = eval("Fake" + b + "V2()")
        cqc = transpile(qc, backend)
        
        for i in range(7):  
            result = backend.run(cqc, shots=8192).result()
            counts = result.get_counts()
            #avg_fid = avg_fid + hellinger_fidelity(perf_counts, counts)
            fids.append(bench.score(Counter(counts)))
                
        fids.sort()
        print(b + ",", fids[3])
            


execute_on_backends()