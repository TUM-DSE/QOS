#!/usr/bin/python3

from operator import delitem
import sys
import os
from subprocess import Popen, PIPE
import csv

times = 11
results = []

setup = []
encrp = []
# compu = []
# decr = []

vec_sum = 0
scal_sum = 0
speedup_sum = 0


# Source
# IBMQ resource page: https://quantum-computing.ibm.com/services/resources?tab=systems
# More fake backends at: https://qiskit.org/documentation/apidoc/providers_fake_provider.html
backends = {
    # "FakeArmonkV2": 1,
    # "FakeAthensV2": 5,
    "FakeBelemV2": 5,
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
    "FakeCasablancaV2": 7,
    # "FakeJakartaV2": 7,
    # "FakeLagosV2": 7,
    # "FakeMelbourneV2": 14,
    # "FakeGuadalupeV2": 16,
    "FakeAlmadenV2": 20,
    # "FakeBoeblingenV2": 20,
    # "FakeSingaporeV2": 20,
    # "FakeJohannesburgV2": 20,
    "FakeCairoV2": 27,
    # "FakeHanoiV2": 27,
    # "FakeParisV2": 27,
    # "FakeSydneyV2": 27,
    # "FakeTorontoV2": 27,
    # "FakeKolkataV2": 27,
    # "FakeMontrealV2": 27,
    # "FakeCambridgeV2": 28,
    # "FakeWashingtonV2": 127,
}

benchmarks = {
    "HamiltonianSimulationBenchmark": 0,
    "VQEBenchmark": 0,
    "ErrorCorrectionBenchmark": 0,
    "BitCodeBenchmark": 0,
    "PhaseCodeBenchmark": 0,
    "VanillaQAOABenchmark": 0,
    "FermionicSwapQAOABenchmark": 0,
    "GHZBenchmark": 0,
}

runs = 10

for i, j in backends.items():
    for x, y in benchmarks.items():
        # This variable is used to increment the number of qbits of the benchmark.
        # It is used as an exponent of 2.
        n = 2

        while 2**n <= backends[i]:
            print(
                "./main.py",
                str(i),
                str(x),
                str(2**n),
                str(runs),
            )
            p = Popen(
                [
                    "./main.py",
                    str(i),
                    str(x),
                    str(2**n),
                    str(runs),
                    "results/",
                ],
                stdout=PIPE,
            )
            aux = p.stdout.readlines()
            print(aux)
            n += 1
