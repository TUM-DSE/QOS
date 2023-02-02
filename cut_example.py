# %%
from qvm import cut
from qvm.bisection import Bisection
from qiskit import QuantumCircuit
from benchmarks.chemistry import HamiltonianSimulationBenchmark
from benchmarks.chemistry import VQEBenchmark
from benchmarks.quantum_information import GHZBenchmark
from benchmarks.error_correction import BitCodeBenchmark

from qiskit.providers.aer import StatevectorSimulator
from qiskit.providers.fake_provider import FakeTorontoV2
from qvm.prob import ProbDistribution
from qiskit.visualization import plot_histogram
from collections import Counter
from qvm.knit import merge
from qvm.knit import knit
from qvm.frag_executor import FragmentExecutor
from qvm.cut import STANDARD_VIRTUAL_GATES
from benchmarks._utils import fidelity, perfect_counts

# %%
bench = HamiltonianSimulationBenchmark(4)
# bench = BitCodeBenchmark(8, 4)
circuit = bench.circuit()
# print(circuit)
# print(circuit[1])

back = FakeTorontoV2()
# back = StatevectorSimulator()
out = back.run(circuit).result().get_counts()
plot_histogram(out)

# %%
passes = Bisection()
distcircuit = cut.cut(circuit, passes)
frags = distcircuit.fragments
print(frags)
frag1 = distcircuit.fragment_as_circuit(frags[0])
frag2 = distcircuit.fragment_as_circuit(frags[1])
# frag1.draw()

# frag2.draw()
out1 = back.run(frag1).result().get_counts()
prob1 = ProbDistribution.from_counts(out1)
out2 = back.run(frag2).result().get_counts()
prob2 = ProbDistribution.from_counts(out2)

# %%
plot_histogram(out1)
# %%
plot_histogram(out2)

# %%
# executor1 = FragmentExecutor(distcircuit, frags[0])
# executor2 = FragmentExecutor(distcircuit, frags[1])

# final_knit = knit([executor1, executor2], STANDARD_VIRTUAL_GATES)
final_knit = merge([prob1, prob2]).counts()
perf = perfect_counts(circuit)
plot_histogram(perf)
# %%
plot_histogram(final_knit)

# %%
fidelity(perf, final_knit)
