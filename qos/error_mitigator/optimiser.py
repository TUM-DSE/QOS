from typing import Any, Dict, List
from abc import ABC, abstractmethod
import pdb
from time import sleep
import networkx as nx

from qiskit.circuit.library import Barrier
from qiskit.circuit import *

from qos.error_mitigator.virtualizer import Virtualizer
from qos.error_mitigator.types import TransformationPass
from qos.types.types import Engine, Qernel
#import qos.database as db

from qvm.compiler.virtualization import BisectionPass, OptimalDecompositionPass
from qvm.compiler.virtualization.reduce_deps import CircularDependencyBreaker, GreedyDependencyBreaker, QubitDependencyMinimizer
from qvm.compiler.distr_transpiler import QubitReuser
from qvm.compiler.virtualization.wire_decomp import OptimalWireCutter
from qvm import VirtualCircuit
from qvm.compiler.dag import *
import numpy as np

from FrozenQubits.helper_FrozenQubits import drop_hotspot_node, halt_qubits
from FrozenQubits.helper_qaoa import pqc_QAOA, bind_QAOA, _gen_angles


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

class QubitFreezingPass(TransformationPass):
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
        bisection_pass = BisectionPass(self._size_to_reach)
        vsqs = q.get_virtual_subqernels()

        if len(vsqs) > 0:
            for vsq in vsqs:
                qc = vsq.get_circuit()
                new_circuit = bisection_pass.run(qc, budget)
                #virtual_circuit = VirtualCircuit(new_circuit)
                vsq.set_circuit(new_circuit)
        else:
            qc = q.get_circuit()        
            new_circuit = bisection_pass.run(qc, budget)
            #virtual_circuit = VirtualCircuit(new_circuit)
            sub_qernel = Qernel()
            sub_qernel.set_circuit(new_circuit)    
            sub_qernel.set_metadata(q.get_metadata())
            q.add_virtual_subqernel(sub_qernel)    

        return q
    
    def cost(self, q: Qernel) -> int:
        optimal_bisection_pass = BisectionPass(self._size_to_reach)
        vsqs = q.get_virtual_subqernels()
        cost = 0

        if len(vsqs) > 0:
            highest_cost = 0
            for vsq in vsqs:
                qc = vsq.get_circuit()
                cost = optimal_bisection_pass.get_budget(qc)
                if cost > highest_cost:
                    highest_cost = cost
        else:
            qc = q.get_circuit()        
            cost = optimal_bisection_pass.get_budget(qc) 

        return cost

class GVOptimalDecompositionPass(GateVirtualizationPass):
    _size_to_reach: int

    def __init__(self, size_to_reach: int):
        self._size_to_reach = size_to_reach

    def name(self):
        return "OptimalDecompositionPass"
    
    def run(self, q: Qernel, budget: int) -> Qernel:
        optimal_decomposition_pass = OptimalDecompositionPass(self._size_to_reach)
        vsqs = q.get_virtual_subqernels()

        if len(vsqs) > 0:
            for vsq in vsqs:
                qc = vsq.get_circuit()
                new_circuit = optimal_decomposition_pass.run(qc, budget)
                #virtual_circuit = VirtualCircuit(new_circuit)
                vsq.set_circuit(new_circuit)
        else:
            qc = q.get_circuit()        
            new_circuit = optimal_decomposition_pass.run(qc, budget)
            #virtual_circuit = VirtualCircuit(new_circuit)
            sub_qernel = Qernel()
            sub_qernel.set_circuit(new_circuit)    
            sub_qernel.set_metadata(q.get_metadata())
            q.add_virtual_subqernel(sub_qernel)    

        return q
    
    def cost(self, q: Qernel, final_cost) -> int:
        optimal_decomposition_pass = OptimalDecompositionPass(self._size_to_reach)
        vsqs = q.get_virtual_subqernels()
        cost = 0

        if len(vsqs) > 0:
            highest_cost = 0
            for vsq in vsqs:
                qc = vsq.get_circuit()
                cost = optimal_decomposition_pass.get_budget(qc)
                if cost > highest_cost:
                    highest_cost = cost
        else:
            qc = q.get_circuit()        
            cost = optimal_decomposition_pass.get_budget(qc) 

        final_cost.value = cost

        return cost

class CircularDependencyBreakerPass(GateVirtualizationPass):
    def name(self):
        return "CircularDependencyBreakerPass"
    
    def run(self, q: Qernel, budget: int) -> Qernel:
        circular_dependency_breaker_pass = CircularDependencyBreaker()
        vsqs = q.get_virtual_subqernels()

        if len(vsqs) > 0:
            for vsq in vsqs:
                qc = vsq.get_circuit()._circuit
                new_circuit = circular_dependency_breaker_pass.run(qc, budget)
                #virtual_circuit = VirtualCircuit(new_circuit)
                vsq.set_circuit(new_circuit)
        else:
            qc = q.get_circuit()        
            new_circuit = circular_dependency_breaker_pass.run(qc, budget)
            #virtual_circuit = VirtualCircuit(new_circuit)
            sub_qernel = Qernel()
            sub_qernel.set_circuit(new_circuit)    
            sub_qernel.set_metadata(q.get_metadata())
            q.add_virtual_subqernel(sub_qernel)    

        return q
    
class GreedyDependencyBreakerPass(GateVirtualizationPass):
    def name(self):
        return "GreedyDependencyBreakerPass"
    
    def run(self, q: Qernel, budget: int) -> Qernel:
        greedy_dependency_breaker_pass = GreedyDependencyBreaker()
        vsqs = q.get_virtual_subqernels()

        if len(vsqs) > 0:
            for vsq in vsqs:
                qc = vsq.get_circuit()._circuit
                new_circuit = greedy_dependency_breaker_pass.run(qc, budget)
                #virtual_circuit = VirtualCircuit(new_circuit)
                vsq.set_circuit(new_circuit)
        else:
            qc = q.get_circuit()        
            new_circuit = greedy_dependency_breaker_pass.run(qc, budget)
            #virtual_circuit = VirtualCircuit(new_circuit)
            sub_qernel = Qernel()
            sub_qernel.set_circuit(new_circuit)    
            sub_qernel.set_metadata(q.get_metadata())
            q.add_virtual_subqernel(sub_qernel)    

        return q
    
class QubitDependencyMinimizerPass(GateVirtualizationPass):
    def name(self):
        return "QubitDependencyMinimizerPass"
    
    def run(self, q: Qernel, budget: int) -> Qernel:
        qubit_dependency_minimizer_pass = QubitDependencyMinimizer()
        vsqs = q.get_virtual_subqernels()

        if len(vsqs) > 0:
            for vsq in vsqs:
                qc = vsq.get_circuit()._circuit
                new_circuit = qubit_dependency_minimizer_pass.run(qc, budget)
                #virtual_circuit = VirtualCircuit(new_circuit)
                vsq.set_circuit(new_circuit)
        else:
            qc = q.get_circuit()        
            new_circuit = qubit_dependency_minimizer_pass.run(qc, budget)
            #virtual_circuit = VirtualCircuit(new_circuit)
            sub_qernel = Qernel()
            sub_qernel.set_circuit(new_circuit)    
            sub_qernel.set_metadata(q.get_metadata())
            q.add_virtual_subqernel(sub_qernel)    

        return q
    
class RandomQubitReusePass(QubitReusePass):
    _size_to_reach: int

    def __init__(self, size_to_reach: int):
        self._size_to_reach = size_to_reach
    
    def name(self):
        return "RandomQubitReusePass"
    
    def run(self, q: Qernel) -> Qernel:
        random_qubit_reuser_pass = QubitReuser(self._size_to_reach)
        vsqs = q.get_virtual_subqernels()
        #sqs = q.get_subqernels()

        if len(vsqs) > 0:
            for vsq in vsqs:
                qc = vsq.get_circuit()
                virtual_circuit = VirtualCircuit(qc)
                random_qubit_reuser_pass.run(virtual_circuit)
                vsq.set_circuit(virtual_circuit)
        else:
            qc = q.get_circuit()
            virtual_circuit = VirtualCircuit(qc)        
            random_qubit_reuser_pass.run(virtual_circuit)
            sub_qernel = Qernel()
            sub_qernel.set_circuit(virtual_circuit)    
            sub_qernel.set_metadata(q.get_metadata())
            q.add_virtual_subqernel(sub_qernel)    

        return q

class OptimalWireCuttingPass(WireCuttingPass):
    _size_to_reach: int

    def __init__(self, size_to_reach: int):
        self._size_to_reach = size_to_reach

    def name(self):
        return "OptimalWireCuttingPass"
    
    def run(self, q: Qernel, budget: int) -> Qernel:
        optimal_wire_cutting_pass = OptimalWireCutter(self._size_to_reach)
        vsqs = q.get_virtual_subqernels()

        if len(vsqs) > 0:
            for vsq in vsqs:
                qc = vsq.get_circuit()
                new_circuit = optimal_wire_cutting_pass.run(qc, budget)

                """                
                vgates = []
                for d in new_circuit.data:
                    label = d.operation.label
                    if label is None:
                        continue
                    if 'v' in label:
                        vgates.append(d)

                vgate_pairs = {}

                for vg in vgates:
                    label = vg.operation.label
                    parts = label.split('_')
                    suffix = parts[-2]
                    if suffix in vgate_pairs:
                        vgate_pairs[suffix].append(vg)
                    else:
                        vgate_pairs[suffix] = [vg]
 
                for vg_no, vg_list in vgate_pairs.items():
                    op_name = vg_list[0].operation.label.split("_")[2]
                    op_0 = vg_list[0].qubits[0]
                    op_1 = vg_list[1].qubits[0]
                    new_instr = CircuitInstruction(operation=VIRTUAL_GATE_TYPES[op_name](Instruction(name=op_name, num_qubits=2, num_clbits=0, params=[])), qubits=(op_0, op_1), clbits=())
                    new_circuit.data.remove(vg_list[0])
                    new_circuit.data.remove(vg_list[1])
                    new_circuit.data.append(new_instr)
                                   
                print(new_circuit)
                virtual_circuit = VirtualCircuit(new_circuit)
                print(virtual_circuit._circuit)
                """
                vsq.set_circuit(new_circuit)
        else:
            qc = q.get_circuit()        
            new_circuit = optimal_wire_cutting_pass.run(qc, budget)
            #virtual_circuit = VirtualCircuit(new_circuit)
            sub_qernel = Qernel()
            sub_qernel.set_circuit(new_circuit)    
            sub_qernel.set_metadata(q.get_metadata())
            q.add_virtual_subqernel(sub_qernel)    

        return q
    
    def cost(self, q: Qernel, final_cost) -> int:
        optimal_wire_cutting_pass = OptimalWireCutter(self._size_to_reach)
        vsqs = q.get_virtual_subqernels()
        cost = 0

        if len(vsqs) > 0:
            highest_cost = 0
            for vsq in vsqs:
                qc = vsq.get_circuit()
                cost = optimal_wire_cutting_pass.get_budget(qc)
                if cost > highest_cost:
                    highest_cost = cost
        else:
            qc = q.get_circuit()        
            cost = optimal_wire_cutting_pass.get_budget(qc) 
        
        final_cost.value = cost

        return cost

class FrozenQubitsPass(QubitFreezingPass):
    _qubits_to_freeze: int

    def __init__(self, qubits_to_freeze: int):
        self._qubits_to_freeze = qubits_to_freeze

    def name(self):
        return "FrozenQubitsPass"
    
    def run(self, q: Qernel) -> Qernel:
        circuit = q.get_circuit()
        metadata = q.get_metadata()
        h = metadata['h']
        J = metadata['J']
        offset = metadata['offset']
        num_layers = metadata['num_layers']

        G = nx.Graph()
        G.add_edges_from(list(J.keys()))
        G.add_nodes_from(list(h.keys()))

        list_of_halting_qubits=[]
        for i in range(self._qubits_to_freeze):
            G, list_of_halting_qubits = drop_hotspot_node(G, list_of_fixed_vars=list_of_halting_qubits, verbosity=0)

        sub_Ising_list = halt_qubits(J=J, h=h, offset=offset, halting_list=list_of_halting_qubits)
     
        for sub_problem in sub_Ising_list:
            new_QAOA = pqc_QAOA(J=sub_problem['J'], h=sub_problem['h'], num_layers=num_layers)
            new_circuit = new_QAOA['qc']
            #beta, gamma =_gen_angles(new_circuit, sub_problem['J'])
            gamma = np.random.uniform(0, 2 * np.pi, 1)[0]
            beta = np.random.uniform(0, np.pi, 1)[0]
            new_circuit = bind_QAOA(new_circuit, new_QAOA['params'], beta, gamma)
            #virtual_circuit = VirtualCircuit(new_circuit)
            sub_qernel = Qernel()
            sub_qernel.set_circuit(new_circuit)
            qaoa_metadata = {
                "h" : sub_problem['h'],
                "J" : sub_problem['J'],
                "offset" : sub_problem['offset'],
                "num_layers" : 1,
                "num_clbits": new_circuit.num_clbits
            }
            sub_qernel.set_metadata(qaoa_metadata)
            q.add_virtual_subqernel(sub_qernel)    

        return q