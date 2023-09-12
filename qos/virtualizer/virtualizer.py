from typing import Any, Dict, List
from abc import ABC, abstractmethod
import copy
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
        new_sub_qernels = []

        for i in range(len(sub_qernels)):
            sub_kernel = sub_qernels.pop()
            vqc = sub_kernel.get_circuit()
            if isinstance(vqc, VirtualCircuit):
                for frag, frag_circuit in vqc.fragment_circuits.items():
                    instance_labels = vqc.get_instance_labels(frag)
                    instantiations = generate_instantiations(frag_circuit, instance_labels)
                    for c in instantiations:
                        new_qernel = Qernel()
                        new_qernel.set_circuit(c)
                        new_sub_qernels.append(new_qernel)                        
                        to_return.append(new_qernel)

        for nsq in new_sub_qernels:
            qernel.add_subqernel(nsq)
        
        return to_return
    
    def results():
        return None

