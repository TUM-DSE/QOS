from abc import ABC, abstractmethod
from typing import Optional


class Operation(ABC):
    @abstractmethod
    def qasm(self) -> str:
        pass


class UnaryOperation(Operation, ABC):
    def __init__(self, name: str, qubit: int) -> None:
        self.name = name
        self.qubit = qubit


class BinaryOperation(Operation, ABC):
    def __init__(self, name: str, qubit1: int, qubit2: int) -> None:
        self.name = name
        self.qubit1 = qubit1
        self.qubit2 = qubit2


class UnaryGate(UnaryOperation):
    def __init__(
        self, name: str, qubit: int, params: Optional[list[float]] = None
    ) -> None:
        super().__init__(name, qubit)
        if params is None:
            params = []
        self.params = params

    def qasm(self) -> str:
        if len(self.params) == 0:
            return f"{self.name} q[{self.qubit}]"
        else:
            return f"{self.name}({','.join([str(p) for p in self.params])}) q[{self.qubit}]"


class BinaryGate(BinaryOperation):
    def __init__(
        self, name: str, qubit1: int, qubit2: int, params: Optional[list[float]] = None
    ) -> None:
        super().__init__(name, qubit1, qubit2)
        if params is None:
            params = []
        self.params = params

    def qasm(self) -> str:
        if len(self.params) == 0:
            return f"{self.name} q[{self.qubit1}], q[{self.qubit2}]"
        else:
            return f"{self.name}({','.join([str(p) for p in self.params])}) q[{self.qubit1}], q[{self.qubit2}]"


class Measurement(UnaryOperation):
    def __init__(self, qubit: int, clbit: int) -> None:
        super().__init__("measure", qubit)
        self.clbit = clbit

    def qasm(self) -> str:
        return f"measure q[{self.qubit}] -> c[{self.clbit}]"


class UnaryBarrier(UnaryOperation):
    def __init__(self, qubit: int) -> None:
        super().__init__("barrier", qubit)

    def qasm(self) -> str:
        return f"barrier q[{self.qubit}]"


class BinaryBarrier(BinaryOperation):
    def __init__(self, qubit1: int, qubit2: int) -> None:
        super().__init__("barrier", qubit1, qubit2)

    def qasm(self) -> str:
        return f"barrier q[{self.qubit1}], q[{self.qubit2}]"
