from typing import Any, Dict, List
from abc import ABC, abstractmethod
import pdb
from time import sleep

from qos.virtualizer.virtualizer import Virtualizer
from qos.distributed_transpiler.types import TransformationPass
from qos.types import Engine, Qernel
import qos.database as db
from qos.tools import debugPrint
from qvm.qvm.compiler.virtualization import BisectionPass, OptimalDecompositionPass
from qvm.qvm.compiler.virtualization.reduce_deps import CircularDependencyBreaker, GreedyDependencyBreaker, QubitDependencyMinimizer
from qvm.qvm.compiler.distr_transpiler import QubitReuser
from qvm.qvm.compiler.virtualization.wire_decomp import OptimalWireCutter
from qvm.qvm import VirtualCircuit


class Optimiser(Engine):
    def __init__(self) -> None:
        pass

    def run(self, qernel: Qernel) -> int:

        # Here the Optimizer would do its job, but for now we just copy the qernel and adds it as a subqernel to the original
        new_qernel = Qernel()
        new_qernel.circuit = qernel.circuit
        new_qernel.provider = qernel.provider
        new_id = db.addQernel(new_qernel)
        new_qernel.id = new_id

        # This is the list of transformations that have been applied to the qernel in the order they were applied
        qernel.transformations = []
        db.setQernelField(qernel.id, "transformations", str(qernel.transformations))

        # This would be an example of a case where the transformer didnt modify the qernel and the subqernel would just be the same as the inital qernel
        qernel.subqernels.append(new_qernel)
        db.addSubqernel(qernel.id, new_qernel.id)

        #Virtualizer = Virtualizer()
        #Virtualizer.submit(qernel)

        # print("----------------------------Transformer: ")
        # debugPrint()
        # pdb.set_trace()

        return qernel
    
    def results(self) -> None:
        pass

class GateVirtualizationPass(TransformationPass):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def run(self, q: Qernel):
        pass

class WireCuttingPass(TransformationPass):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def run(self, q: Qernel):
        pass

class QubitReusePass(TransformationPass):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def run(self, q: Qernel):
        pass

class GVBisectionPass(GateVirtualizationPass):
    _size_to_reach: int

    def __init__(self, size_to_reach: int):
        self._size_to_reach = size_to_reach

    def name(self):
        return "BisectionPass"
    
    def run(self, q: Qernel, budget: int) -> Qernel:
        circuit = q.get_circuit()
        bisection_pass = BisectionPass(self._size_to_reach)
        new_circuit = bisection_pass.run(circuit, budget)
        virtual_circuit = VirtualCircuit(new_circuit)
        sub_qernel = Qernel()
        sub_qernel.set_circuit(virtual_circuit)    
        q.add_subqernel(sub_qernel)    

        return q

class GVOptimalDecompositionPass(GateVirtualizationPass):
    _size_to_reach: int

    def __init__(self, size_to_reach: int):
        self._size_to_reach = size_to_reach

    def name(self):
        return "OptimalDecompositionPass"
    
    def run(self, q: Qernel, budget: int) -> Qernel:
        circuit = q.get_circuit()
        optimal_decomposition_pass = OptimalDecompositionPass(self._size_to_reach)
        new_circuit = optimal_decomposition_pass.run(circuit, budget)
        virtual_circuit = VirtualCircuit(new_circuit)
        sub_qernel = Qernel()
        sub_qernel.set_circuit(virtual_circuit)    
        q.add_subqernel(sub_qernel)    

        return q

class CircularDependencyBreakerPass(GateVirtualizationPass):
    def name(self):
        return "CircularDependencyBreakerPass"
    
    def run(self, q: Qernel, budget: int) -> Qernel:
        circuit = q.get_circuit()
        circular_dependency_breaker_pass = CircularDependencyBreaker()
        new_circuit = circular_dependency_breaker_pass.run(circuit, budget)
        virtual_circuit = VirtualCircuit(new_circuit)
        sub_qernel = Qernel()
        sub_qernel.set_circuit(virtual_circuit)    
        q.add_subqernel(sub_qernel)    

        return q
    
class GreedyDependencyBreakerPass(GateVirtualizationPass):
    def name(self):
        return "GreedyDependencyBreakerPass"
    
    def run(self, q: Qernel, budget: int) -> Qernel:
        circuit = q.get_circuit()
        greedy_dependency_breaker_pass = GreedyDependencyBreaker()
        new_circuit = greedy_dependency_breaker_pass.run(circuit, budget)
        virtual_circuit = VirtualCircuit(new_circuit)
        sub_qernel = Qernel()
        sub_qernel.set_circuit(virtual_circuit)    
        q.add_subqernel(sub_qernel)    

        return q
    
class QubitDependencyMinimizerPass(GateVirtualizationPass):
    def name(self):
        return "QubitDependencyMinimizerPass"
    
    def run(self, q: Qernel, budget: int) -> Qernel:
        circuit = q.get_circuit()
        qubit_dependency_minimizer_pass = QubitDependencyMinimizer()
        new_circuit = qubit_dependency_minimizer_pass.run(circuit, budget)
        virtual_circuit = VirtualCircuit(new_circuit)
        sub_qernel = Qernel()
        sub_qernel.set_circuit(virtual_circuit)    
        q.add_subqernel(sub_qernel)    

        return q
    
class RandomQubitReusePass(QubitReusePass):
    _size_to_reach: int

    def __init__(self, size_to_reach: int):
        self._size_to_reach = size_to_reach
    
    def name(self):
        return "RandomQubitReusePass"
    
    def run(self, q: Qernel) -> Qernel:
        virtual_circuit = q.get_circuit()
        random_qubit_reuser_pass = QubitReuser(self._size_to_reach)

        if not isinstance(virtual_circuit, VirtualCircuit):
            virtual_circuit = VirtualCircuit(virtual_circuit)
        
        random_qubit_reuser_pass.run(virtual_circuit)

        sub_qernel = Qernel()
        sub_qernel.set_circuit(virtual_circuit)    
        q.add_subqernel(sub_qernel)

        return q

class OptimalWireCuttingPass(WireCuttingPass):
    _size_to_reach: int

    def __init__(self, size_to_reach: int):
        self._size_to_reach = size_to_reach

    def name(self):
        return "OptimalWireCuttingPass"
    
    def run(self, q: Qernel, budget: int) -> Qernel:
        circuit = q.get_circuit()
        optimal_wire_cutting_pass = OptimalWireCutter(self._size_to_reach)
        new_circuit = optimal_wire_cutting_pass.run(circuit, budget)
        virtual_circuit = VirtualCircuit(new_circuit)
        sub_qernel = Qernel()
        sub_qernel.set_circuit(virtual_circuit)    
        q.add_subqernel(sub_qernel)    

        return q