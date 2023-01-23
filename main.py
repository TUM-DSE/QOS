from qiskit.providers import fake_provider

from benchmarks import chemistry
from benchmarks import error_correction
from benchmarks import optimization
from benchmarks import quantum_information

benchmarks = {
    "HamiltonianSimulationBenchmark": chemistry.HamiltonianSimulationBenchmark,
    "VQEBenchmark": chemistry.VQEBenchmark,
    "ErrorCorrectionBenchmark": error_correction.ErrorCorrectionBenchmark,
    "BitCodeBenchmark": error_correction.BitCodeBenchmark,
    "PhaseCodeBenchmark": error_correction.PhaseCodeBenchmark,
    "VanillaQAOABenchmark": optimization.VanillaQAOABenchmark,
    "FermionicSwapQAOABenchmark": optimization.FermionicSwapQAOABenchmark,
    "GHZBenchmark": quantum_information.GHZBenchmark,
}

# Source
# IBMQ resource page: https://quantum-computing.ibm.com/services/resources?tab=systems
# More fake backends at: https://qiskit.org/documentation/apidoc/providers_fake_provider.html
backends = {
    "sim_armonk_1": fake_provider.FakeArmonkV2,
    "sim_athens_5": fake_provider.FakeAthensV2,
    "sim_belem_5": fake_provider.FakeBelemV2,
    "sim_yorktown_5": fake_provider.FakeYorktownV2,
    "sim_bogota_5": fake_provider.FakeBogotaV2,
    "sim_ourense_5": fake_provider.FakeOurenseV2,
    "sim_valencia_5": fake_provider.FakeValenciaV2,
    "sim_burlington_5": fake_provider.FakeBurlingtonV2,
    "sim_essex_5": fake_provider.FakeEssexV2,
    "sim_rome_5": fake_provider.FakeRomeV2,
    "sim_manila_5": fake_provider.FakeManilaV2,
    "sim_lima_5": fake_provider.FakeLimaV2,
    "sim_london_5": fake_provider.FakeLondonV2,
    "sim_vigo_5": fake_provider.FakeVigoV2,
    "sim_casablanca_7": fake_provider.FakeCasablancaV2,
    "sim_jakarta_7": fake_provider.FakeJakartaV2,
    "sim_lagos_7": fake_provider.FakeLagosV2,
    "sim_melbourne_14": fake_provider.FakeMelbourneV2,
    "sim_guadalupe_16": fake_provider.FakeGuadalupeV2,
    "sim_almaden_20": fake_provider.FakeAlmadenV2,
    "sim_boeblingen_20": fake_provider.FakeBoeblingenV2,
    "sim_singapore_20": fake_provider.FakeSingaporeV2,
    "sim_johannesburg_20": fake_provider.FakeJohannesburgV2,
    "sim_cairo_27": fake_provider.FakeCairoV2,
    "sim_hanoi_27": fake_provider.FakeHanoiV2,
    "sim_paris_27": fake_provider.FakeParisV2,
    "sim_sydney_27": fake_provider.FakeSydneyV2,
    "sim_toronto_27": fake_provider.FakeTorontoV2,
    "sim_kolkata_27": fake_provider.FakeKolkataV2,
    "sim_montreal_27": fake_provider.FakeMontrealV2,
    "sim_cambridge_28": fake_provider.FakeCambridgeV2,
    "sim_cambridge_127": fake_provider.FakeWashingtonV2,
    # "sim_poughkeepsie_5": fake_provider.FakePoughkeepsieV2,  # qubits??
    # "sim_brooklyn_5": fake_provider.FakeBrooklynV2,  # qubits??
    # "sim_rochester_5": fake_provider.FakeRochesterV2,  # qubits??
    # "sim_mumbai_5": fake_provider.FakeMumbaiV2,  # qubits??
    # "sim_santiago_5": fake_provider.FakeSantiagoV2,  # qubits??
    # "sim_manhattan_28": fake_provider.FakeManhattanV2,  # qubits??
}

print("===================")
print("Bechmark list:")
for i, j in enumerate(benchmarks):
    print(i, j)

print("-------------------")
print("Backend list:")
for i, j in enumerate(backends):
    print(i, j)


bench_selection = input("Benchmark[0.." + str(len(benchmarks) - 1) + "]: ")
back_selection = input("Backend[0.." + str(len(backends) - 1) + "]: ")


circuit = benchmarks[bench_selection]
