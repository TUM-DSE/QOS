import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2

import qvm

from fid import calculate_fidelity


def main():
    circuit = EfficientSU2(8, entanglement="linear", reps=2).decompose()
    circuit.measure_all()
    circuit = circuit.bind_parameters(
        {param: np.random.randn() / 2 for param in circuit.parameters}
    )

    print(circuit)
    comp = qvm.CutterCompiler(size_to_reach=4)
    virtual_circuit = comp.run(circuit, budget=2)
    for frag in virtual_circuit.fragment_circuits.values():
        print(frag)
    result, _ = qvm.run_virtual_circuit(virtual_circuit, shots=10000)
    print(calculate_fidelity(circuit, result))


if __name__ == "__main__":
    main()
