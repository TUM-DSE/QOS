import numpy as np

#from qiskit.quantum_info.analysis import hellinger_fidelity
from qiskit.compiler.transpile import * 
from qiskit.providers.fake_provider import *

from qstack.benchmarking import (
    BitCodeBenchmark,
    GHZBenchmark,
    HamiltonianSimulationBenchmark,
    PhaseCodeBenchmark,
    VanillaQAOABenchmark,
    VQEBenchmark,
)

from qstack.qos.merge_engine import (
    MergeEngine
)

from qstack.qos.backends import (
    IBMQQPU
)

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

qpu = IBMQQPU(backend.name())

merge_engine = MergeEngine(qpu)

circuits = []

num_circuits = 30

for x in range(0, num_circuits): 
    circuits.append(GHZBenchmark(np.random.randint(10) + 1).circuit())
    circuits[x] = transpile(circuits[x], backend, optimization_level=3)   

circ1 = circuits.pop(0)
job1 = backend.run(circ1)
counts1 = job1.result().get_counts()

avrg = 0
maxFid = 0

max_tests = 10

for x in range(0, max_tests):
    circ2 = circuits[np.random.randint(max - 1)]

    circ3 = merge_engine.merge_qernels(circ1, circ2, forced=True)
    
    job2 = backend.run(circ3)
    counts2 = job2.result().get_counts()
    counts3 = get_counts(counts1, counts2)    

    oldScore = circ1.score(counts3)

    circ2 = merge_engine.find_best_match(circ1, circuits)
    
    circ3 = merge_engine.merge_qernels(circ1, circ2, forced=True)
    
    job2 = backend.run(circ3)
    counts2 = job2.result().get_counts()
    
    counts3 = get_counts(counts1, counts2)
               
    newScore = circ1.score(counts3)

    if (newScore - oldScore) > maxFid:
        maxFid = newScore - oldScore
    
    avrg = avrg + (newScore - oldScore)

avrg = avrg / max_tests
print(avrg * 100)
print(maxFid * 100)