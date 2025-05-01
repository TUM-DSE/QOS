from abc import ABC, abstractmethod
from typing import Any, Dict, List
import sys
from qos.backends.types import QPU
from qvm.qvm.virtual_circuit import VirtualCircuit
from qvm.qvm.quasi_distr import *
from qiskit import dagcircuit, QuantumCircuit
from ..dag import DAG
#from qos.distributed_transpiler.types import AnalysisPass
#from qos.distributed_transpiler.types import TransformationPass


# This should contain any kind of circuit, qiskit circuit or circ, etc
#class QCircuit(ABC):
#    id: int
#    type: str
#    args: Dict[str, Any]
#    _circuit: str
#
#    def __init__(self) -> None:
#        self.args = {}
#        self._circuit = None

class Transformations:
    # This is an Abstract class for the optimisation transformations

    @abstractmethod
    def submit():
        pass

    @abstractmethod
    def results():
        pass

class Qernel(ABC):
    id: int

    #qpu: QPU
    status: str
    metadata: Dict[str, Any]
    results: Dict[str, Any] | QuasiDistr
    submit_time: float
    parent: Any #In case this is a subquernel this will store the parent qernel object
    local_queue: List[tuple] #The local queue will store a tuple of estimated execution time and the time when the circuit was submitted
    src_qernels: List[tuple] #List[tuple(Qernel,int)], int is the number of qbits for that fragment
    waiting_time: float
    #status: str
    #analysis: List[AnalysisPass]
    #transformations: List[TransformationPass]
    
    # * This is the circuit provider that was submitted by the client
    # * so we can return the result using the same API
    provider: str
    circuit: QuantumCircuit | VirtualCircuit
    dag = DAG
    matching: List[tuple]
    match: tuple
    args: Dict[str, Any]

    def __init__(self, qc: QuantumCircuit | VirtualCircuit = None, metadata: dict[str, Any] = None) -> None:
        self.args = {}

        # If a Qernel has subqernels it wont have dependencies and vice-versa.
        # This is because if is has subqernels then it means that it was an original circuit
        #   that was split by the Optimizer
        # If it has dependencies it means that it was a new system-created Qernel created
        #   by the Multiprogrammer to encapsulate a merged circuit and its original Qernel
        #   dependencies
        self.submit_time = -1
        self.provider = ""
        self.circuit = qc
        if qc is not None:
            self.dag = DAG(qc)
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata
        self.matching = []
        self.match = None
        self.subqernels: List[Qernel] = []
        self.virtual_subqernels: List[Qernel] = []
        self.dependencies: List[Qernel] = []
        self.args["status"] = "PENDING"
        self.src_qernels = []

    def set_metadata(self, metadata):
        self.metadata = metadata

    def is_bundle(self) -> bool:
        return self.src_qernels != []

    def get_metadata(self) -> dict[str, Any]:
        return self.metadata
    
    def edit_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata.update(metadata)
    
    def set_circuit(self, qc: QuantumCircuit | VirtualCircuit) -> None:
        self.circuit = qc

    def get_circuit(self) -> QuantumCircuit:
        if self.circuit is None:
            return self.dag.to_circuit()
        else:
            return self.circuit
        
    def append_circuit(self, qc: QuantumCircuit) -> None:
        toReturn = QuantumCircuit( self.circuit.num_qubits + qc.num_qubits, self.circuit.num_clbits + qc.num_clbits)
        qubits1 = [*range(0, self.circuit.num_qubits)]
        clbits1 = [*range(0, self.circuit.num_clbits)]
        qubits2 = [*range(self.circuit.num_qubits, self.circuit.num_qubits + qc.num_qubits)]
        clbits2 = [*range(self.circuit.num_clbits, self.circuit.num_clbits + qc.num_clbits)]
    
        toReturn.compose(self.circuit, qubits=qubits1, clbits=clbits1, inplace=True)
        toReturn.compose(qc, qubits=qubits2, clbits=clbits2, inplace=True)
    
        self.circuit = toReturn
        
    def get_dag(self) -> DAG:
        return self.dag
      
    def add_subqernel(self, q) -> None:
        self.subqernels.append(q)
        q.parent = self

    def add_virtual_subqernel(self, q) -> None:
        self.virtual_subqernels.append(q)

    def add_virtual_subqernel(self, q) -> None:
        self.virtual_subqernels.append(q)

    def get_subqernels(self):
        return self.subqernels
    
    def get_virtual_subqernels(self):
        vsqs = []

        for sq in self.subqernels:
            if isinstance(sq.get_circuit(), VirtualCircuit):
                vsqs.append(sq)
        
        return vsqs
    
    def get_virtual_subqernels(self):
        return self.virtual_subqernels
    
    def get_results(self):
        return self.results

    def set_results(self, results: Dict[str, Any] | QuasiDistr):
        self.results = results

    # TODO - Doesnt work, needs to be fixed and added a few more properties
    def __format__(self, __format_spec: str) -> str:
        return (
            "Qernel id: "
            + str(self.id)
            + "\n\t status: \t"
            + self.status
            + "\n"
            + "\n\t #subqernels: "
            + len(self.subqernels)
            + "\n"
        )


class Engine(ABC):
    """Generic interface for implementations of QOS Engines"""

    @abstractmethod
    def run(self, args: Dict[str, Any]) -> int:
        pass

    @abstractmethod
    def results(self, id: int, args: Dict[str, Any]) -> None:
        pass


class Backend(ABC):
    @abstractmethod
    def run(self, args: Dict[str, Any]) -> int:
        pass


class scheduler_policy(ABC):
    @abstractmethod
    def schedule(self, newQernel: Qernel, kargs: Dict):
        pass
