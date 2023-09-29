import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.providers.fake_provider import *

from qvm.qvm.compiler.virtualization.wire_decomp import OptimalWireCutter
from qvm.qvm.compiler.virtualization.gate_decomp import BisectionPass
from qvm.qvm import run_virtual_circuit, VirtualCircuit
from qvm.examples.fid import calculate_fidelity


def main():
    #circuit = EfficientSU2(6, reps=2).decompose()
    #clreg = ClassicalRegister(6)
    #circuit.add_register(clreg)
    #circuit.measure(range(6), range(6))
    #circuit = circuit.bind_parameters(
        #{param: np.random.randn() / 2 for param in circuit.parameters}
    #)
    nqubits = 16
    circuit = QuantumCircuit(nqubits, nqubits)
    circuit.h(0)
    for i in range(1, nqubits):
        circuit.cx(i - 1, i)

    circuit.measure_all()
    circuit = RemoveBarriers()(circuit)

    print(circuit.draw())

    cp = circuit.copy()
    cut_circuit = cp
    gv_pass = BisectionPass(8)
    cut_circuit = gv_pass.run(circuit, 10)
    print(cut_circuit.draw())
    #comp_pass = OptimalWireCutter(5)
    #cut_circuit = comp_pass.run(cut_circuit, 10)
    #cut_circuit.measure_all()
    #print(cut_circuit.draw())
    #cp.measure_all()

    virt_circ = VirtualCircuit(cut_circuit)

    result, _ = run_virtual_circuit(virt_circ, shots=20000)
    #backend = FakeGuadalupe()
    #counts = backend.run()
    print(calculate_fidelity(cp, result))


if __name__ == "__main__":
    main()
