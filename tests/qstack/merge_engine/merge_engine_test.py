import numpy as np
import sys
from collections import Counter

sys.path.insert(0, '/mnt/c/Users/giort/Documents/GitHub/qos/')

#from qiskit.quantum_info.analysis import hellinger_fidelity
from qiskit.compiler import transpile  

from qiskit.providers.fake_provider import *

from qstack.benchmarking import (
    BitCodeBenchmark,
    GHZBenchmark,
    HamiltonianSimulationBenchmark,
    PhaseCodeBenchmark,
    VanillaQAOABenchmark,
    VQEBenchmark,
)

from qstack.qos.merge_engine import MergeEngine

from qstack.backends import IBMQQPU

def get_counts(counts1, counts2):
    kl = len(list(counts1.keys())[0])
    counts3 = {}
    
    for (key, value) in counts2.items():
        newKey = key[:kl]
        counts3.update({newKey : 0})
        for (key2, value2) in counts2.items():
            if newKey == key2[:kl]:                
                counts3[newKey] = counts3[newKey] + value2
    return counts3
    

provider = FakeProvider()

backends = [x for x in provider.backends() if hasattr(x.configuration(), 'processor_type') and x.configuration().n_qubits > 20]
backend = backends[np.random.randint(len(backends))]

#print(backend.backend_name)
qpu = IBMQQPU("FakeWashingtonV2")

merge_engine = MergeEngine(qpu)

circuits = []
benchs = []

num_circuits = 1000

for x in range(0, num_circuits):
    rnd = np.random.randint(4)
    if rnd == 0:
        benchs.append(HamiltonianSimulationBenchmark(num_qubits=np.random.randint(10) + 1, total_time=np.random.randint(10) + 1))
    elif rnd == 1:
        benchs.append(GHZBenchmark(num_qubits=np.random.randint(10) + 1))
    elif rnd == 2:
        benchs.append(VanillaQAOABenchmark(num_qubits=np.random.randint(10) + 1))
    elif rnd == 3:
        benchs.append(VQEBenchmark(num_qubits=np.random.randint(10) + 1))
    elif rnd == 4:
        benchs.append(ErrorCorrectionBenchmark(num_data_qubits=np.random.randint(10) + 1, num_correction_measurement_rounds=np.random.randint(5) + 1))
    circuits.append(benchs[x].circuit())
    #circuits[x] = transpile(circuits[x], backend, optimization_level=3)   

circ1 = circuits.pop(0)
transpile(circ1, backend, optimization_level=3)
job1 = backend.run(circ1)
counts1 = job1.result().get_counts()
oldScore0 = benchs[0].score(Counter(counts1))
#print("original: ", oldScore)

avrg1 = 0
avrg2 = 0
maxFid = 0
index = 0

max_tests = 50

for x in range(0, max_tests):
    index = np.random.randint(num_circuits - 1)
    circ2 = circuits[index]

    circ3 = merge_engine.merge_qernels(circ1, circ2, forced=True)
    transpile(circ3, backend, optimization_level=3)
    
    job2 = backend.run(circ3)
    counts2 = job2.result().get_counts()
    counts3 = Counter(get_counts(counts1, counts2))    

    oldScore = benchs[0].score(counts3)
    
    circ2 = merge_engine.find_best_match(circ1, circuits)
    
    circ3 = merge_engine.merge_qernels(circ1, circ2, forced=True)
    transpile(circ3, backend, optimization_level=3)
    
    job2 = backend.run(circ3)
    counts2 = job2.result().get_counts()
    
    counts3 = Counter(get_counts(counts1, counts2))
               
    newScore = benchs[index].score(counts3)
    print("new2: ", newScore)

    if (newScore - oldScore) > maxFid:
        maxFid = newScore - oldScore
    
    print((newScore - oldScore) * 100)
    print(" ")
    avrg1 = avrg1 + (newScore - oldScore0)
    avrg2 = avrg2 + (newScore - oldScore)

avrg1 = avrg1 / max_tests
avrg2 = avrg2 / max_tests
print("Compared to solo: ", avrg1 * 100)
print("Random vs best match: ", avrg2 * 100)
print(maxFid * 100)