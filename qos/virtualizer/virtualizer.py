from typing import Any, Dict, List
from abc import ABC, abstractmethod
import copy
from multiprocessing import Pool

from qos.types import Engine, Qernel
import qos.database as db
import pdb
from time import sleep
from qiskit.circuit import QuantumCircuit

from qvm.qvm.virtual_circuit import VirtualCircuit, generate_instantiations
from qvm.qvm.quasi_distr import *
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

        for vsq in virtual_sub_qernels:
            instantiated_sub_qernel = Qernel(vsq.get_circuit()._circuit)
            qernel.add_subqernel(instantiated_sub_qernel)
            vqc = vsq.get_circuit()
            assert isinstance(vqc, VirtualCircuit)
            for frag, frag_circuit in vqc.fragment_circuits.items():
                new_vqernel = Qernel()
                new_vqernel.set_circuit(frag)
                new_vqernel.edit_metadata(vsq.get_metadata())               

                instance_labels = vqc.get_instance_labels(frag)
                instantiations = generate_instantiations(frag_circuit, instance_labels)

                new_vqernel.edit_metadata({"num_instantiations": len(instantiations)})
                vsq.add_virtual_subqernel(new_vqernel)

                for c in instantiations:
                    new_qernel = Qernel()
                    new_qernel.set_circuit(c)
                    new_qernel.edit_metadata(vsq.get_metadata())
                    instantiated_sub_qernel.add_subqernel(new_qernel)   
        
        return qernel
    
    def results():
        return None


class GVKnitter(Knitter):
    def run(self, qernel: Qernel) -> None:
        #subqernels = qernel.get_subqernels()[0].get_subqernels()
        #root_virtual_subqernel = qernel.get_virtual_subqernels()[0]
        #virtual_subqernels = root_virtual_subqernel.get_virtual_subqernels()
        virtual_subqernels = qernel.get_virtual_subqernels()

        for i, vsq in enumerate(virtual_subqernels):
            subqernels = qernel.get_subqernels()[i].get_subqernels()
            results = {}
            tmp_results = []

            counter = 0
            virtual_child_subqernels = vsq.get_virtual_subqernels()

            for vcsq in virtual_child_subqernels:
                num_instantiations = vcsq.get_metadata()["num_instantiations"]
                for j in range(num_instantiations):
                    tmp_results.append(QuasiDistr.from_counts(subqernels[counter].get_results()))
                    counter = counter + 1
                results[vcsq.get_circuit()] = tmp_results

                tmp_results = []

            clbits = vsq.get_metadata()["num_clbits"]
            shots = vsq.get_metadata()["shots"]

            with Pool() as pool:
               vsq.set_results(vsq.get_circuit().knit(results, pool).to_counts(clbits, shots))

        #results = {}
        #tmp_results = []
        """"
        counter = 0
        for vsq in virtual_subqernels:
            num_instantiations = vsq.get_metadata()["num_instantiations"]
            for i in range(num_instantiations):
                tmp_results.append(QuasiDistr.from_counts(subqernels[counter].get_results()))
                counter = counter + 1
            results[vsq.get_circuit()] = tmp_results
            tmp_results = []

        clbits = qernel.get_metadata()["num_clbits"]
        shots = qernel.get_metadata()["shots"]

        with Pool() as pool:
            to_return = root_virtual_subqernel.get_circuit().knit(results, pool).to_counts(clbits, shots)
        
        return to_return
        """
    
    def results():
        return None
