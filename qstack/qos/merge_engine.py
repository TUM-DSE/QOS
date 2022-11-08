from typing import List

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qstack.qernel.qernel import Qernel, QernelArgs
from qstack.types import QPUWrapper


class MergeEngine:
    """QOS engine for merging qernels based on a matching score"""
    
    _qpu = None
    _backendCNOTerror = 0
    _backendReadoutError = 0
    
    def __init__(self, qpu : QPUWrapper) -> None:
        self._qpu = qpu
        self._backendCNOTerror = self._qpu.properties().gate_error('cx',(1,0))
        self._backendReadoutError = self._qpu.properties().readout_error(0)
     
    def __merge_qernels(self, q1: Qernel, q2: Qernel) -> Qernel:
        toReturn = Qernel(q1.num_qubits + q2.num_qubits, q1.num_clbits + q2.num_clbits)
        qubits1 = [*range(0, q1.num_qubits)]
        clbits1 = [*range(0, q1.num_clbits)]
        qubits2 = [*range(q1.num_qubits, q1.num_qubits + q2.num_qubits)]
        clbits2 = [*range(q1.num_clbits, q1.num_clbits + q2.num_clbits)]

        toReturn.compose(q1, qubits=qubits1, clbits=clbits1, inplace=True)
        toReturn.compose(q2, qubits=qubits2, clbits=clbits2, inplace=True)
        
        return toReturn
    
    def merge_qernels(self, q1: Qernel, q2: Qernel, forced: bool = False, simThres: float = 0.8, costThres: float = 0.3) -> Qernel:
        """ Merge Qernels if compatible or if forced=True"""
        if forced is True:
            return self.__merge_qernels(q1, q2)
        else:
            score = self.get_matching_score(q1, q2)
            if score >= simThres:
                q3 = self.__merge_qernels(q1, q2)
                id = self._qpu.register_qernel(q3)
                cost = self._qpu.cost(id)
                if cost <= costThres:
                    return q3
            return None
    
    def find_best_match(self, q1: Qernel, qernels: List, simThres: float = 0.8) -> Qernel:
        max = 0
        toReturn = None
        
        for q2 in qernels:
            score = self.get_matching_score(q1, q2)
            if score >= simThres and score > max:
                max = score
                toReturn = q2
        
        return toReturn
            
    
    def get_matching_score(self, q1: Qernel, q2: Qernel) -> float:
        """ Get a matching score in [0,1] for two Qernels"""
        score = 1.0
    
        depthDiff = self.__depthComparison(q1, q2)
        score = score - 0.2 * depthDiff
        
        cnotDiff = self.__cnotComparison(q1, q2)
        score = score - 0.4 * cnotDiff
        
        measurementDiff = self.__measurementComparison(q1, q2)
        score = score - 0.4 * measurementDiff
        
        return score 
        
    def __depthComparison(q1: Qernel, q2: Qernel) -> float:
        depthDiff = q1.depth() / q2.depth()
        
        depthDiffPerc = depthDiff if depthDiff >= 1 else 1 / depthDiff
                   
        return depthDiffPerc


    def __cnotComparison(q1: Qernel, q2: Qernel) -> float:
        ops1 = q1.count_ops()
        ops2 = q2.count_ops()
        nCNOTs1, nCNOTs2 = 0,0
        
        for (key, value) in ops1.items():
            if key == 'cx':
                nCNOTs1 = value
                
        for (key, value) in ops2.items():
            if key == 'cx':
                nCNOTs2 = value
        
        CNOTdensity = (nCNOTs1 + nCNOTs2) / (q1.width() + q2.width())
        
        return self._backendCNOTerror * CNOTdensity 

    def __measurementComparison(q1: Qernel, q2: Qernel) -> float:
        ops1 = q1.count_ops()
        ops2 = q2.count_ops()
        nMs1, nMs2 = 0,0
        
        for (key, value) in ops1.items():
            if key == 'measure':
                nMs1 = value
                
        for (key, value) in ops2.items():
            if key == 'measure':
                nMs2 = value
        
        measurementDensity = (nMs1 + nMs2 - q1.num_qubits + q2.num_qubits) / (q1.width() + q2.width())
        
        return self._backendReadoutError * measurementDensity
