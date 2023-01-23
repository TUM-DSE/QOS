from dataclasses import dataclass
from .circuit import Circuit
from .operation import UnaryOperation, BinaryOperation


class UnaryPlaceholder(UnaryOperation):
    def __init__(self, name: str, qubit: int) -> None:
        super().__init__(f"{name}_placeholder", qubit)

    def qasm(self) -> str:
        return f"{self.name} q[{self.qubit}]"


class BinaryPlaceholder(BinaryOperation):
    def __init__(self, name: str, qubit1: int, qubit2: int) -> None:
        super().__init__(f"{name}_placeholder", qubit1, qubit2)

    def qasm(self) -> str:
        return f"{self.name} q[{self.qubit1}], q[{self.qubit2}]"


@dataclass
class Arguments:
    params: dict[str, list[float]]
    subcircs: dict[str, Circuit]


class Qernel(Circuit):
    """Qernel is a quantum circuit that is can hold arguments such as placeholders and parameters"""

    def insert_arguments(self, args: Arguments) -> Circuit:
        pass
