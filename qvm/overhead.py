from qiskit import QuantumCircuit

from vqc.virtual_gate import VirtualBinaryGate


def virtualization_overhead(circuit: QuantumCircuit) -> int:
    vgates = [
        instr.operation
        for instr in circuit.data
        if isinstance(instr.operation, VirtualBinaryGate)
    ]
    overhead = 1
    for op in vgates:
        overhead *= len(op.configure())
    return overhead
