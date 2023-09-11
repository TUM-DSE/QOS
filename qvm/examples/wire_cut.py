import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit import QuantumCircuit, ClassicalRegister

from qvm.compiler.virtualization.wire_decomp import OptimalWireCutter
from qvm import run_virtual_circuit, VirtualCircuit
from fid import calculate_fidelity


def main():
    circuit = EfficientSU2(6, reps=2).decompose()
    clreg = ClassicalRegister(6)
    circuit.add_register(clreg)
    circuit.measure(range(6), range(6))
    circuit = circuit.bind_parameters(
        {param: np.random.randn() / 2 for param in circuit.parameters}
    )
    print(circuit.draw())

    cp = circuit.copy()
    comp_pass = OptimalWireCutter(4)
    cut_circuit = comp_pass.run(circuit, 2)
    print(cut_circuit.draw())

    virt_circ = VirtualCircuit(cut_circuit)

    result, _ = run_virtual_circuit(virt_circ, shots=10000)
    print(calculate_fidelity(cp, result))


if __name__ == "__main__":
    main()
