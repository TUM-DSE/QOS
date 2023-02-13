from operator import delitem
import sys
import os
from subprocess import Popen, PIPE
import subprocess
import csv
import math
import pdb

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
}

shots = 8192
qbits = [
    [[12], [12]],
    [[6, 6], [6, 6]],
    [[4, 4, 4], [4, 4, 4]],
]  # This is for adding other combinations.
rounds = 3
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


def how_many_qbits(initial_qbits, benchmark):
    if (
        benchmark == "HamiltonianSimulationBenchmark"
        or benchmark == "GHZBenchmark"
        or benchmark == "VanillaQAOABenchmark"
    ):
        return initial_qbits
    else:
        return initial_qbits // 2


list_benchmarks = list(benchmarks.keys())
this = [unique_combinations(list_benchmarks, len(i)) for i in qbits]
this = [len(i) for i in this]
total_ids = sum(this)

# pdb.set_trace()
for i in qbits:
    combinations = unique_combinations(list_benchmarks, len(i))
    for j in combinations:
        f = open("configs/config_" + str(id) + ".yml", "w")
        f.write("config:\n")
        f.write("  path: results/\n")
        f.write("  static: false\n")
        f.write("  nshots: " + str(shots) + "\n")
        f.write("  benchmarks:\n")
        for idx in range(len(i[0])):
            f.write("    - name: " + j[0] + "\n")
            f.write("      nqbits: " + str(how_many_qbits(i[0][idx], j[0])) + "\n")
            if benchmarks[j[0]] == 2:
                f.write("      rounds: " + str(rounds) + "\n")
            else:
                f.write("      rounds:\n")
            f.write("      cut: false\n")
            f.write("      frags:\n")
            f.write("        - backend: " + backend + "\n")
        for idx in range(len(i[1])):
            f.write("    - name: " + j[1] + "\n")
            f.write("      nqbits: " + str(how_many_qbits(i[1][idx], j[1])) + "\n")
            if benchmarks[j[1]] == 2:
                f.write("      rounds: " + str(rounds) + "\n")
            else:
                f.write("      rounds:\n")
            f.write("      cut: false\n")
            f.write("      frags:\n")
            f.write("        - backend: " + backend + "\n")
        id += 1


if sys.argv[1] == "run":
    for i in range(total_ids):
        this = subprocess.run(["python3", "main.py", "configs/config_" + str(i)])

if sys.argv[1] == "clean":
    for i in range(total_ids):
        this = subprocess.run(["rm", "configs/config_*"])
