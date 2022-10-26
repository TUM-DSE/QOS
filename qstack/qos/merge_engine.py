from qiskit.circuit.quantumcircuit import QuantumCircuit
from qstack.qernel.qernel import Qernel, QernelArgs
from qstack.types import QPUWrapper


class MergeEngine:
    """QOS engine for merging qernels based on a matching score"""
    
    _qpu = None
    
    def __init__(self, qpu : QPUWrapper = None) -> None:
        self._qpu = qpu
        
    def merge_qernels(self, qernel1: Qernel, qernel2: Qernel, qpu: QPUWrapper = None, forced: bool = False) -> Qernel:
    """ Merge Qernels if compatible or if forced=True"""
        pass
    
    def get_matching_score(self, qernel1: Qernel, qernel2: Qernel, qpu: QPUWrapper = None) -> float:
    """ Get a matching score in [0,1] for two Qernels"""
        pass
