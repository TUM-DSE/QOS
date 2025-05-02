from qiskit.circuit import QuantumCircuit

from qvm.virtual_gates import VirtualBinaryGate


def num_virtual_gates(circuit: QuantumCircuit) -> int:
    return sum(1 for instr in circuit if isinstance(instr.operation, VirtualBinaryGate))
