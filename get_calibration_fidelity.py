import os
import shutil
from datetime import datetime, date, timedelta
import json
import time
import numpy as np

from qiskit_ibm_provider import IBMProvider
from qiskit.quantum_info.analysis import hellinger_fidelity
from qiskit.providers.fake_provider import *
from qiskit import transpile
from qiskit_aer import AerSimulator

from benchmarks.circuits import *
from benchmarks.plot.plot import * 

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
    provider = IBMProvider(token='87ae595a5a0b9624fe36f477550700ee4b4dc540061a89951f197a0cd36d639e2c5e6307d533993123eaa925d9bea2de14a02b659219646ea4750e1768c76bf1')
    backend = provider.get_backend(machinename)
    
    start_date = datetime(day=1, month=1, year=2023, hour=10)
    end_date = datetime(day=10, month=7, year=2023, hour=10) 

    delta = end_date - start_date

    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)

        try:
            properties = backend.properties(datetime=day)
        except:
            continue
        
        if properties is None:
            continue
            
        properties = properties.to_dict()

        convert_dict_to_json(properties, "calibration_data/" + machinename + "/" +datetime_to_str(day) + ".json")

        time.sleep(0.5)

def move_and_rename_file(src_path, dst_dir, new_name):
    dst_path = os.path.join(dst_dir, new_name)
    # print(dst_path)
    # Move the file to the destination path
    shutil.copy2(src_path, dst_path)

def iterate_files_in_directory():
    bench = get_circuits("ghz", (6,7))[0]
    simulator = AerSimulator()

    perf_results = simulator.run(bench, shots=8192).result().get_counts()

    #b = "ibmq_kolkata"
    
    directory = "calibration_data/ibm_perth/"
        
    fids = []

    files = os.listdir(directory)

    files.sort()

    for file in files:
        filename = os.fsdecode(file)

        file_path = directory + filename

        if os.path.isfile(file_path):
            move_and_rename_file(file_path, "/home/manosgior/Documents/qos-venv/lib/python3.10/site-packages/qiskit/providers/fake_provider/backends/perth", "props_perth.json")
            backend = FakePerth() 
                    
            tmp_fids = []
            for i in range(3):
                trans_qc = transpile(bench, backend)

                result = backend.run(trans_qc, shots=8192).result()
                counts = result.get_counts()    
                fid = hellinger_fidelity(counts, perf_results)

                tmp_fids.append(fid)        

            fids.append(np.mean(tmp_fids))
            print(min(tmp_fids))

    with open("perth_fids_6.txt", 'w') as f:
        for fid in fids:
            f.write(str(fid) + "\n")

#get_callibration_data("ibm_perth")
#iterate_files_in_directory()
csv_file_path = "results/large_circuits_solo_"
dataframes = []

for i in [4, 8, 12, 16, 20, 24]:
    dataframes.append(pd.read_csv(csv_file_path + str(i) + ".csv"))

dataframes.append(pd.read_csv("results/spatial_hetero_12.csv"))
dataframes.append(pd.read_csv("results/perth_fids_6.csv"))

custom_plot_scal_hetero_challenges(dataframes, ["(a) Scalability", "(b) Spatial Heterogeneity", "(c) Temporal Variance"], ["Fidelity"], ["Number of Qubits", "QPU", "Calibration Day"])