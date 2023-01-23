from typing import Optional
from .application_layer.qernel.operation import (
    BinaryGate,
    Operation,
    UnaryGate,
    UnaryOperation,
    BinaryOperation,
    Measurement,
    UnaryBarrier,
    BinaryBarrier,
)

from qiskit.circuit import QuantumCircuit


class Circuit:
    """A simple static circuit as a list of operations"""

    _ops: list[Operation]
    _num_qubits: int
    _num_clbits: int

    def __init__(
        self,
        operations: list[Operation],
        num_qubits: Optional[int] = None,
        num_clbits: Optional[int] = None,
    ) -> None:
        self._ops = operations
        if num_qubits is None:
            self._num_qubits = 0
        if num_clbits is None:
            self._num_clbits = 0

        for op in operations:
            if isinstance(op, UnaryOperation):
                self._num_qubits = max(self._num_qubits, op.qubit + 1)
            elif isinstance(op, BinaryOperation):
                self._num_qubits = max(self._num_qubits, op.qubit1 + 1, op.qubit2 + 1)
            elif isinstance(op, Measurement):
                self._num_qubits = max(self._num_qubits, op.qubit + 1)
                self._num_clbits = max(self._num_clbits, op.clbit + 1)

    @property
    def operations(self) -> list[Operation]:
        return self._ops.copy()

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_clbits(self) -> int:
        return self._num_clbits

    @classmethod
    def from_qiskit_circuit(cls, circuit: QuantumCircuit) -> "Circuit":
        ops: list[Operation] = []
        for qs_instr in circuit.data:
            if qs_instr[0].name == "measure":
                ops.append(
                    Measurement(qs_instr.qubits[0].index, qs_instr.qubits[0].index)
                )
            elif qs_instr[0].name == "barrier":
                if len(qs_instr.qubits) == 1:
                    ops.append(UnaryBarrier(qs_instr.qubits[0].index))
                elif len(qs_instr.qubits) == 2:
                    ops.append(
                        BinaryBarrier(
                            qs_instr.qubits[0].index, qs_instr.qubits[1].index
                        )
                    )
                else:
                    raise ValueError("Barrier with more than 2 qubits not supported")
            else:
                if len(qs_instr.qubits) == 1:
                    qubit = circuit.find_bit(qs_instr.qubits[0]).index
                    ops.append(
                        UnaryGate(
                            qs_instr.operation.name, qubit, qs_instr.operation.params
                        )
                    )
                elif len(qs_instr.qubits) == 2:
                    qubit1 = circuit.find_bit(qs_instr.qubits[0]).index
                    qubit2 = circuit.find_bit(qs_instr.qubits[1]).index
                    ops.append(
                        BinaryGate(
                            qs_instr.operation.name,
                            qubit1,
                            qubit2,
                            qs_instr.operation.params,
                        )
                    )
                else:
                    raise ValueError("Gate with more than 2 qubits not supported")
        return Circuit(ops, circuit.num_qubits, circuit.num_clbits)

    @classmethod
    def from_qasm(self, qasm: str) -> "Circuit":
        """Return a Circuit from a QASM representation"""
        return Circuit.from_qiskit_circuit(QuantumCircuit.from_qasm_str(qasm))

    def to_qiskit_circuit(self) -> QuantumCircuit:
        return QuantumCircuit.from_qasm_str(self.qasm())

    def qasm(self) -> str:
        """Return a QASM representation of the circuit"""
        qasm = "OPENQASM 2.0;\n"
        qasm += 'include "qelib1.inc";\n\n'
        qasm += f"qreg q[{self._num_qubits}];\n"
        qasm += f"creg c[{self._num_clbits}];\n\n"
        for op in self._ops:
            qasm += f"{op.qasm()};\n"
        return qasm
