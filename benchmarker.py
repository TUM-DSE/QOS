from operator import delitem
import sys
import os
from subprocess import Popen, PIPE
import subprocess
import csv
import math

# Source
# IBMQ resource page: https://quantum-computing.ibm.com/services/resources?tab=systems
# More fake backends at: https://qiskit.org/documentation/apidoc/providers_fake_provider.html
backends = {
    # "FakeArmonkV2": 1,
    # "FakeAthensV2": 5,
    # "FakeBelemV2": 5,
    # "FakeYorktownV2": 5,
    # "FakeBogotaV2": 5,
    # "FakeOurenseV2": 5,
    # "FakeValenciaV2": 5,
    # "FakeBurlingtonV2": 5,
    # "FakeEssexV2": 5,
    # "FakeRomeV2": 5,
    # "FakeManilaV2": 5,
    # "FakeLimaV2": 5,
    # "FakeLondonV2": 5,
    # "FakeVigoV2": 5,
    # "FakeCasablancaV2": 7,
    # "FakeAlmaden": 7,
    # "FakeJakartaV2": 7,
    # "FakeLagosV2": 7,
    # "FakeMelbourneV2": 14,
    # "FakeGuadalupeV2": 16,
    # "FakeAlmadenV2": 20,
    # "FakeBoeblingenV2": 20,
    # "FakeSingaporeV2": 20,
    # "FakeJohannesburgV2": 20,
    # "FakeCairoV2": 27,
    # "FakeHanoiV2": 27,
    # "FakeParisV2": 27,
    # "FakeSydneyV2": 27,
    # "FakeTorontoV2": 27,
    # "FakeKolkataV2": 27,
    # "FakeMontrealV2": 27,
    # "FakeCambridgeV2": 28,
    "FakeWashingtonV2": 127,
}

benchmarks = {
    "HamiltonianSimulationBenchmark": 1,
    "VQEBenchmark": 1,
    "VanillaQAOABenchmark": 1,
    "GHZBenchmark": 1,
    "BitCodeBenchmark": 2,
    "PhaseCodeBenchmark": 2,
    "MerminBellBenchmark": 1,
    "FermionicSwapQAOABenchmark": 1,
}

shots = 8192
qbits = [[16]]  # This is for adding other combinations.
rounds = 1
runs = 1
# qbits = [0.25, 0.5, 0.75, 1]

id = 0

backend = "FakeTorontoV2"

if sys.argv[1] == "gen":
    # run_cmd = "python main.py -backend {} -benchmarks {} -runs {} -shots {} -bits {}"
    run_cmd = "python main.py"


def unique_combinations(lst, length):
    def get_combinations(lst, length, start, comb, result):
        if length == 0:
            result.append(list(comb))
            return
        for i in range(start, len(lst)):
            comb.append(lst[i])
            get_combinations(lst, length - 1, i, comb, result)
            comb.pop()

    result = []
    get_combinations(lst, length, 0, [], result)
    return result


list_benchmarks = list(benchmarks.keys())
this = [unique_combinations(list_benchmarks, len(i)) for i in qbits]
this = [len(i) for i in this]
total_ids = sum(this)
static = True

for i in qbits:
    combinations = unique_combinations(list_benchmarks, len(i))
    for j in combinations:
        f = open("configs/config_" + str(id) + ".yml", "w")
        f.write("config:\n")
        f.write("  path: results/\n")
        f.write("  static: " + str(static) +"\n")
        f.write("  nshots: " + str(shots) + "\n")
        f.write("  nruns: " + str(runs) + "\n")
        f.write("  benchmarks:\n")
        for idx in range(len(j)):
            f.write("    - name: " + j[idx] + "\n")
            if j[idx] == "VQEBenchmark" or j[idx] == "BitCodeBenchmark" or j[idx] == "PhaseCodeBenchmark":
                f.write("      nqbits: " + str(int(i[idx] / 2)) + "\n")
            else:
                f.write("      nqbits: " + str(i[idx]) + "\n")
            f.write("      nlayers: " + "\n")
            f.write("      time_step: " + "\n")
            f.write("      total_time: " + "\n")
            f.write("      initial_state: " + "\n")
            if benchmarks[j[idx]] == 2:
                f.write("      rounds: " + str(rounds) + "\n")
            else:
                f.write("      rounds:\n")
            f.write("      cut: false\n")
            f.write("      frags:\n")
            f.write("        - backend: " + backend + "\n")
        id += 1

#exit()
if sys.argv[1] == "run":
    for i in range(total_ids):
        this = subprocess.run(["python3", "main.py", "configs/config_" + str(i)])

# create another input option to clean all files inside the configs folder
if sys.argv[1] == "clean":
    for i in range(total_ids):
        os.remove("configs/config_" + str(i) + ".yml")
