from dataclasses import dataclass
from typing import Dict, Union

from qiskit.circuit import QuantumCircuit, Gate, Parameter


class UnaryPlaceholder(Gate):
    def __init__(self, name: str) -> None:
        super().__init__(name, 1, [])


class BinaryPlaceholder(Gate):
    def __init__(self, name: str) -> None:
        super().__init__(name, 2, [])


@dataclass
class Input:
    params: Dict[Parameter, float]
    subcircuits: Dict[Union[UnaryPlaceholder, BinaryPlaceholder], QuantumCircuit]


class Qernel(QuantumCircuit):
    @staticmethod
    def from_circuit(circuit: QuantumCircuit) -> "Qernel":
        qernel = Qernel(
            *circuit.qregs,
            *circuit.cregs,
            name=circuit.name,
            global_phase=circuit.global_phase,
            metadata=circuit.metadata
        )
        for instr in circuit.data:
            qernel.append(instr)
        return qernel

    def with_inserted_inputs(self, input: Input) -> "Qernel":
        # TODO
        pass
