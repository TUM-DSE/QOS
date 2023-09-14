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
    def run(self, qernel: Qernel) -> Any:
        pass
    
    @abstractmethod
    def results(self, qernel: Qernel):
        pass

class Instantiator(Virtualizer):
    @abstractmethod
    def run(self, qernel: Qernel) -> list[Qernel]:
        pass
    
    @abstractmethod
    def results(self, qernel: Qernel):
        pass

class Knitter(Virtualizer):
    @abstractmethod
    def run(self, qernel: Qernel) -> dict[str, int]:
        pass
    
    @abstractmethod
    def results(self, qernel: Qernel):
        pass

class GVInstatiator(Instantiator):
    
    def run(self, qernel: Qernel) -> list[Qernel]:
        qc = qernel.get_circuit()
        virtual_sub_qernels = qernel.get_virtual_subqernels()
        instantiated_sub_qernel = Qernel()
        qernel.add_subqernel(instantiated_sub_qernel)

        for vsq in virtual_sub_qernels:
            vqc = vsq.get_circuit()
            assert isinstance(vqc, VirtualCircuit)
            for frag, frag_circuit in vqc.fragment_circuits.items():
                new_qernel = Qernel()
                new_qernel.set_circuit(frag)
                new_qernel.edit_metadata(vsq.get_metadata())
                vsq.add_virtual_subqernel(new_qernel)

                instance_labels = vqc.get_instance_labels(frag)
                instantiations = generate_instantiations(frag_circuit, instance_labels)
                for c in instantiations:
                    new_qernel = Qernel()
                    new_qernel.set_circuit(c)
                    new_qernel.edit_metadata(vsq.get_metadata())
                    instantiated_sub_qernel.add_subqernel(new_qernel)                        

        
        return instantiated_sub_qernel.get_subqernels()
    
    def results():
        return None


class GVKnitter(Knitter):
    def run(self, qernel: Qernel) -> dict[str, int]:
        v_qc = qernel.get_circuit()
        assert isinstance(v_qc, VirtualCircuit)


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
