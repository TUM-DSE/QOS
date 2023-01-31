# %%
from qvm import cut
from qvm.bisection import Bisection
from qiskit import QuantumCircuit
from benchmarks.chemistry import HamiltonianSimulationBenchmark

# %%
bench = HamiltonianSimulationBenchmark(4)
circuit = bench.circuit()
circuit.draw()

# %%
passes = Bisection()

# %%
distcircuit = cut.cut(circuit, passes)

# %%
frags = distcircuit.fragments
print(frags)
frag1 = distcircuit.fragment_as_circuit(frags[0])
frag2 = distcircuit.fragment_as_circuit(frags[1])
# %%
frag1.draw()

# %%
frag2.draw()
