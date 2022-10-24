from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

from qiskit.circuit.quantumcircuit import (
    Bit,
    Instruction,
    InstructionSet,
    Parameter,
    ParameterValueType,
    QuantumCircuit,
    Qubit,
    Register,
)


@dataclass
class QernelArgs:
    """Input to a Qernel"""

    params: Dict[Parameter, float]
    subcircs: Dict[str, QuantumCircuit]


class UnaryPlaceholder(Instruction):
    """A unary placeholder to insert to a qernel"""

    def __init__(self, name: str, clbit: bool = False):
        num_clbits = 1 if clbit else 0
        super().__init__(name=name, num_qubits=1, num_clbits=num_clbits, params=[])

    def _define(self):
        self._definition = QuantumCircuit(1, name=self.name)


class Qernel(QuantumCircuit):
    # placeholder name to instruction indices

    def __init__(
        self,
        *regs: Union[Register, int, Sequence[Bit]],
        name: Optional[str] = None,
        global_phase: ParameterValueType = 0,
        metadata: Optional[Dict] = None,
    ):
        super().__init__(*regs, name=name, global_phase=global_phase, metadata=metadata)

    def placeholder(
        self,
        name: str,
        qubit: Union[int, Qubit],
        clbit: Optional[Union[int, Qubit]] = None,
    ) -> InstructionSet:

        if clbit is not None:
            return self.append(
                UnaryPlaceholder(name=name, clbit=True), [qubit], [clbit]
            )
        else:
            return self.append(UnaryPlaceholder(name=name), [qubit], [])

    def with_input(self, args: QernelArgs) -> QuantumCircuit:
        res_circ = QuantumCircuit(*self.qregs, *self.cregs, name=self.name)
        for instr in self.data:
            if isinstance(instr, UnaryPlaceholder):
                if instr.name not in args.subcircs:
                    raise ValueError(
                        f"Placeholder {instr.name} is not present in input"
                    )
                subcirc = args.subcircs[instr.name]
                if subcirc.num_qubits != 1 or subcirc.num_clbits > 1:
                    raise ValueError(
                        "Subcircuit must have 1 qubit and 0 or 1 clbits, not {} qubits and {} clbits".format(
                            subcirc.num_qubits, subcirc.num_clbits
                        )
                    )
                res_circ.append(subcirc.to_instruction(), instr.qubits, instr.clbits)
            else:
                res_circ.append(instr)

        return res_circ.bind_parameters(args.params)
