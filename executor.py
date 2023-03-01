from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import hellinger_fidelity
import os
import shutil
from qiskit.providers.fake_provider import FakeLagosV2
from benchmarks import *
from collections import Counter

def move_and_rename_file(src_path, dst_dir, new_name):
    dst_path = os.path.join(dst_dir, new_name)
    #print(dst_path)
    # Move the file to the destination path
    shutil.copy2(src_path, dst_path)



def iterate_files_in_directory(dir_path):    
    bench = BitCodeBenchmark(4, 3)
    qc = bench.circuit()
    
    perf_counts = {'0000000' : 4096, '1111111' : 4096}

    for filename in os.listdir(dir_path):
        # Construct the full path to the file by joining the directory path and the filename
        file_path = os.path.join(dir_path, filename)
        #print(file_path)
        
        # Check if the path is a file (as opposed to a directory or a symlink)
        if os.path.isfile(file_path):
            move_and_rename_file(file_path, "/mnt/c/Users/giort/Documents/GitHub/ME/lib/python3.9/site-packages/qiskit/test/mock/backends/lagos", "props_lagos.json")

            backend = FakeLagosV2()
            avg_fid = 0
            
            for i in range (3):
                qc = transpile(qc, backend)
                
                result = backend.run(qc, shots=4096).result()
                counts = result.get_counts()
                #avg_fid = avg_fid + hellinger_fidelity(perf_counts, counts)
                avg_fid = avg_fid + bench.score(Counter(counts))
                
            print(avg_fid / 3)
            

iterate_files_in_directory("/mnt/c/Users/giort/Documents/GitHub/qos/callibration_data")