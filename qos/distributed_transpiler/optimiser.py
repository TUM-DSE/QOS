from typing import Any, Dict, List
from abc import ABC, abstractmethod
import pdb
from time import sleep

from qos.virtualizer.virtualizer import Virtualizer
from qos.distributed_transpiler.types import TransformationPass
from qos.types import Engine, Qernel
import qos.database as db
from qos.tools import debugPrint
from qvm.qvm.compiler.virtualization import BisectionPass
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

class GVBisectionPass(GateVirtualizationPass):
    _size_to_reach: int

    def __init__(self, size_to_reach: int):
        self._size_to_reach = size_to_reach

    def name(self):
        return "BisectionPass"
    
    def run(self, q: Qernel, budget: int) -> Qernel:
        circuit = q.get_circuit()
        virtual_circuit = VirtualCircuit(circuit)

        bisection_pass = BisectionPass(self._size_to_reach)

        new_circuit = bisection_pass.run(circuit, budget)

        return Qernel(new_circuit)




