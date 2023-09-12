from typing import Any, Dict, List
from abc import ABC, abstractmethod
from qos.types import Engine, Qernel
import qos.database as db
import pdb
from time import sleep
from qiskit.circuit import QuantumCircuit

from qvm.qvm.virtual_circuit import VirtualCircuit, generate_instantiations
from qos.tools import debugPrint


class Virtualizer(Engine):
    @abstractmethod
    def run(self, qernel: Qernel) -> int:
        pass
    
    @abstractmethod
    def results(self, qernel: Qernel):
        pass


class GateVirtualizer(Virtualizer):
    
    def run(self, qernel: Qernel) -> list[Qernel]:
        to_return = []
        qc = qernel.get_circuit()
        sub_qernels = qernel.get_subqernels()

        for i in range(len(sub_qernels)):
            vqc = sub_qernels.pop().get_circuit()
            if isinstance(vqc, VirtualCircuit):
                for frag, frag_circuit in vqc.fragment_circuits.items():
                    instance_labels = vqc.get_instance_labels(frag)
                    instantiations = generate_instantiations(frag_circuit, instance_labels)
                    for c in instantiations:
                        new_qernel = Qernel()
                        new_qernel.set_circuit(c)
                        qernel.add_subqernel(new_qernel)
                        to_return.append(new_qernel)

        return to_return
    
    def results():
        return None

