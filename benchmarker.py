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
    "FakeTorontoV2": 27,
    # "FakeKolkataV2": 27,
    # "FakeMontrealV2": 27,
    # "FakeCambridgeV2": 28,
    # "FakeWashingtonV2": 127,
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
qbits = [3, 3]
rounds = 3
# qbits = [0.25, 0.5, 0.75, 1]

id = 0

total_ids = 6 * 6

backend = "FakeTorontoV2"

if sys.argv[1] == "gen":
    # run_cmd = "python main.py -backend {} -benchmarks {} -runs {} -shots {} -bits {}"
    run_cmd = "python main.py"

    for a, b in benchmarks.items():
        for x, y in benchmarks.items():
            f = open("configs/config_" + str(id) + ".yml", "w")
            f.write("config:\n")
            f.write("  path: results/\n")
            f.write("  nshots: " + str(shots) + "\n")
            f.write("  benchmarks:\n")
            f.write("    - name: " + a + "\n")
            f.write("      nqbits: " + str(qbits[0]) + "\n")
            if b == 2:
                f.write("      rounds: " + str(rounds) + "\n")
            else:
                f.write("      rounds:\n")
            f.write("      cut: false\n")
            f.write("      frags:\n")
            f.write("        - backend: " + backend + "\n")
            f.write("    - name: " + x + "\n")
            f.write("      nqbits: " + str(qbits[1]) + "\n")
            if y == 2:
                f.write("      rounds: " + str(rounds) + "\n")
            else:
                f.write("      rounds:\n")
            f.write("      cut: false\n")
            f.write("      frags:\n")
            f.write("        - backend: " + backend + "\n")
            f.close()
            id += 1

if sys.argv[1] == "run":
    for i in range(total_ids):
        this = subprocess.run(["python3", "main.py", "configs/config_" + str(i)])
