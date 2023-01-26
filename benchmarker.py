#!/usr/bin/python3

import subprocess

# Source
# IBMQ resource page: https://quantum-computing.ibm.com/services/resources?tab=systems
# More fake backends at: https://qiskit.org/documentation/apidoc/providers_fake_provider.html
backends = {
    # "FakeArmonkV2": 1,
    # "FakeAthensV2": 5,
    ## "FakeBelemV2": 5,
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
    "FakeJakartaV2": 7,
    # "FakeLagosV2": 7,
    # "FakeMelbourneV2": 14,
    # "FakeGuadalupeV2": 16,
    ##"FakeAlmadenV2": 20,
    # "FakeBoeblingenV2": 20,
    # "FakeSingaporeV2": 20,
    # "FakeJohannesburgV2": 20,
    ##"FakeCairoV2": 27,
    # "FakeHanoiV2": 27,
    # "FakeParisV2": 27,
    # "FakeSydneyV2": 27,
    ##"FakeTorontoV2": 27,
    # "FakeKolkataV2": 27,
    # "FakeMontrealV2": 27,
    # "FakeCambridgeV2": 28,
    # "FakeWashingtonV2": 127,
}

benchmarks = {
    # "HamiltonianSimulationBenchmark": [],
    # "VQEBenchmark": [],
    # "VanillaQAOABenchmark": [],
    "GHZBenchmark": [
        "-bits 4",
    ],
    # "BitCodeBenchmark": [
    #    "-bits 4 -rounds 3",
    #    "-bits 7 -rounds 3",
    #    "-bits 12 -rounds 3",
    #    "-bits 15 -rounds 3",
    #    "-bits 20 -rounds 3",
    #    "-bits 27 -rounds 3",
    # ]
    ##"PhaseCodeBenchmark": ["-rounds 3"],
    # "MerminBellBenchmark": [],
    # "FermionicSwapQAOABenchmark": [],
}

runs = 10
shots = 1024

run_cmd = "./main.py -backend {} -benchmark {} -runs {} -shots {} {}"

for back_name, total_qbits in backends.items():
    for bench_name, bench_args in benchmarks.items():
        for arg in bench_args:
            cmd = run_cmd.format(
                str(back_name), str(bench_name), str(runs), str(shots), arg
            )
            # exit(0)
            print(cmd)
            this = subprocess.getoutput(cmd)
            print(this)
