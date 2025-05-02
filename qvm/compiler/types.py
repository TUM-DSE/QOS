import abc

from qiskit.circuit import QuantumCircuit

from qvm.virtual_circuit import VirtualCircuit


class VirtualizationPass(abc.ABC):
    """A compiler pass that inserts virtual operations into a circuit."""

    @abc.abstractmethod
    def run(self, circuit: QuantumCircuit, budget: int) -> QuantumCircuit:
        pass


class DistributedTranspilerPass(abc.ABC):
    """
    A compiler pass that modifies a virtual circuit (e.g. mapping or qubit reuse).
    """

    @abc.abstractmethod
    def run(self, virt: VirtualCircuit) -> None:
        pass
