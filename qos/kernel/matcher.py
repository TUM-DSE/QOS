from typing import Any, Dict, List
from qos.types import Engine, Qernel
from qos.kernel.multiprogrammer import Multiprogrammer
from qos.kernel.multiprogrammer import pipe_name as multiprog_pipe_name
import qos.database as db
from qiskit.providers.fake_provider import *
import mapomatic as mm
import logging
import pdb
import redis
from qiskit import transpile, QuantumCircuit
import numpy as np
import networkx as nx
import pickle
from networkx.readwrite import json_graph
from qos.dag import DAG
from qiskit_ibm_provider import IBMProvider
from ibm_token import IBM_TOKEN
import mapomatic.layouts as mply

gates = {
    "u3": 1,
    "u2": 1,
    "u1": 1,
    "cx": 2,
    "id": 1,
    "u0": 1,
    "u": 1,
    "p": 1,
    "x": 1,
    "y": 1,
    "z": 1,
    "h": 1,
    "s": 1,
    "sdg": 1,
    "t": 1,
    "tdg": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "sx": 1,
    "sxdg": 1,
    "cz": 1,
    "cy": 1,
    "swap": 1,
    "ch": 1,
    "ccx": 1,
    "cswap": 1,
    "crx": 1,
    "cry": 1,
    "crz": 1,
    "cu1": 1,
    "cp": 1,
    "cu3": 1,
    "csx": 1,
    "cu": 1,
    "rxx": 1,
    "rzz": 1,
    "rccx": 1,
    "rc3x": 1,
    "c3x": 1,
    "c3sqrtx": 1,
    "c4x": 1,
    "ecr": 2,
}


class Matcher(Engine):

    _qpus = []
    _qpu_properties = {}

    def __init__(self, qpus: list = None) -> None:
        max_qpu_id = 0

        max_qpu_id = db.getLastQPUid()

        if qpus == None:
            for i in range(1, max_qpu_id + 1):
                qpu_name = db.getQPU(i).name
                qpu_alias = db.getQPU(i).alias
        
                if "Fake" in qpu_name:
                    backend = eval(qpu_name)()
                    self._qpu_properties[backend.name()] = {}
                    self._qpu_properties[backend.name()][
                    "medianReadoutError"] = self.getMedianReadoutError(backend)
                    basis_gates = backend.configuration().basis_gates
                    for g in basis_gates:
                        self._qpu_properties[backend.name()][g] = self.getMedianGateError(backend, g)
                else:
                    provider = IBMProvider(token=IBM_TOKEN)
                    print("Loading backend {} ({}/{})".format(qpu_alias, i, max_qpu_id))
                    backend = provider.get_backend(qpu_alias)
                    db.setQPUField(i, "backend", pickle.dumps(backend))
                    self._qpu_properties[backend.name] = {}
                    self._qpu_properties[backend.name][
                    "medianReadoutError"] = self.getMedianReadoutError(backend)
                    basis_gates = backend.configuration().basis_gates
                    for g in basis_gates:
                        self._qpu_properties[backend.name][g] = self.getMedianGateError(backend, g)

                self._qpus.append(backend)
        else:
            for q in qpus:
                self._qpus.append(q)

    def getMedianReadoutError(self, backend):
        props = backend.properties()
        readouts = []

        for i in range(backend.configuration().n_qubits):
            readouts.append(props.readout_error(i))

        return np.median(readouts)
    
    def getbackend(self, backend_name:str):

        for i in self._qpus:
            if i.name() == backend_name:
                return i
            
        return None


    def getMedianGateError(self, backend, gate):
        props = backend.properties()
        coupling_map = backend.configuration().coupling_map
        qubits = [i for i in range(backend.configuration().n_qubits)]
        errors = []

        if gate == "reset":
            return 0.0

        if gates[gate] == 1:
            for q in qubits:
                errors.append(props.gate_error(gate, q))
        else:
            for pair in coupling_map:
                errors.append(props.gate_error(gate, pair))

        return np.median(errors)

    def trivialConstFunction(self, circuit: QuantumCircuit, layouts, backend):
        fid = 1.0
        error = 0

        for key, value in circuit.count_ops().items():
            if key == "measure" or key == "barrier":
                continue
            for v in range(value):
                fid *= 1 - self._qpu_properties[backend.name()][key]

        for i in range(circuit.num_qubits):
            fid *= 1 - self._qpu_properties[backend.name()]["medianReadoutError"]

        error = 1 - fid

        return [(layouts[0], error)]

    def accurate_cost_func(self, circ, layouts, backend):
        out = []
        props = backend.properties()
        dt = backend.configuration().dt
        num_qubits = backend.configuration().num_qubits

        t1s = [props.qubit_property(qq, "T1")[0] for qq in range(num_qubits)]
        t2s = [props.qubit_property(qq, "T2")[0] for qq in range(num_qubits)]

        for layout in layouts:
            sch_circ = transpile(
                circ,
                backend,
                initial_layout=layout,
                optimization_level=3,
                scheduling_method="alap",
            )
            error = 0
            fid = 1
            touched = set()
            for item in sch_circ._data:
                if item[0].name == "cx":
                    q0 = sch_circ.find_bit(item[1][0]).index
                    q1 = sch_circ.find_bit(item[1][1]).index
                    fid *= 1 - props.gate_error("cx", [q0, q1])
                    touched.add(q0)
                    touched.add(q1)

                elif item[0].name in ["sx", "x"]:
                    q0 = sch_circ.find_bit(item[1][0]).index
                    fid *= 1 - props.gate_error(item[0].name, q0)
                    touched.add(q0)

                elif item[0].name == "measure":
                    q0 = sch_circ.find_bit(item[1][0]).index
                    fid *= 1 - props.readout_error(q0)
                    touched.add(q0)

                elif item[0].name == "delay":
                    q0 = sch_circ.find_bit(item[1][0]).index
                    if q0 in touched:
                        time = item[0].duration * dt
                        fid *= 1 - self.idle_error(time, t1s[q0], t2s[q0])

            error = 1 - fid
            out.append((layout, error))

        return out

    def idle_error(self, time, t1, t2):
        t2 = min(t1, t2)
        rate1 = 1 / t1
        rate2 = 1 / t2
        p_reset = 1 - np.exp(-time * rate1)
        p_z = (1 - p_reset) * (1 - np.exp(-time * (rate2 - rate1))) / 2
        return p_z + p_reset
    
    def best_overall_layoutv2(self, circuit, backends, successors=True, cost_function=None):
        #pdb.set_trace()

        if not isinstance(backends, list):
            backends = [backends]

        if cost_function is None:
            cost_function = mply.default_cost

        best_out = []

        for backend in backends:
            config = backend.configuration()

            try:
                trans_qc = transpile(circuit, backend, optimization_level=3)
            except NameError as e:
                print("[ERROR] - Can't transpile circuit on backend {}".format(backend.name()))
                return 1

            circ = mm.deflate_circuit(trans_qc)

            circ_qubits = circ.num_qubits
            circuit_gates = set(circ.count_ops()).difference({'barrier', 'reset', 'measure'})
            if not circuit_gates.issubset(backend.configuration().basis_gates):
                continue
            num_qubits = config.num_qubits
            if not config.simulator and circ_qubits <= num_qubits:
                layouts = mply.matching_layouts(circ, config.coupling_map)
                layout_and_error = mply.evaluate_layouts(circ, layouts, backend,
                                                    cost_function=cost_function)
                if any(layout_and_error):
                    layout = layout_and_error[0][0]
                    error = layout_and_error[0][1]
                    best_out.append((layout, config.backend_name, error))
        best_out.sort(key=lambda x: x[2])
        if successors:
            return best_out
        if best_out:
            return best_out[0]
        return best_out
    

    def best_overall_layoutv3(self, circuit, backends, successors=True, cost_function=None):
        #pdb.set_trace()

        if not isinstance(backends, list):
            backends = [backends]

        if cost_function is None:
            cost_function = mply.default_cost

        best_out = []

        try:
            trans_qc27 = transpile(circuit, backend, optimization_level=3)
        except NameError as e:
            print("[ERROR] - Can't transpile circuit on backend {}".format(backend.name()))
            return 1

        try:
            trans_qc127 = transpile(circuit, backend, optimization_level=3)
        except NameError as e:
            print("[ERROR] - Can't transpile circuit on backend {}".format(backend.name()))
            return 1

        circ27 = mm.deflate_circuit(trans_qc27)
        circ127 = mm.deflate_circuit(trans_qc127)

        for backend in backends:
            config = backend.configuration()

            if backend.n_qubits == 27:
                circ = circ27
            elif backend.n_qubits == 127:
                circ = circ127

            circ_qubits = circ.num_qubits
            circuit_gates = set(circ.count_ops()).difference({'barrier', 'reset', 'measure'})
            if not circuit_gates.issubset(backend.configuration().basis_gates):
                continue
            num_qubits = config.num_qubits
            if not config.simulator and circ_qubits <= num_qubits:
                layouts = mply.matching_layouts(circ, config.coupling_map)
                layout_and_error = mply.evaluate_layouts(circ, layouts, backend,
                                                    cost_function=cost_function)
                if any(layout_and_error):
                    layout = layout_and_error[0][0]
                    error = layout_and_error[0][1]
                    best_out.append((layout, config.backend_name, error))
        best_out.sort(key=lambda x: x[2])
        if successors:
            return best_out
        if best_out:
            return best_out[0]
        return best_out

    def match(self, circuit: QuantumCircuit, cost_function=None) -> List:

        #logger = logging.getLogger(__name__)
        #logging.basicConfig(level=10)

        #this = mm.best_overall_layout(circuit, self._qpus, successors=True, cost_function=cost_function
        #)

        if cost_function == None:
            this = self.best_overall_layoutv2(circuit, self._qpus, successors=True, cost_function=self.accurate_cost_func)

        #if cost_function == None:
        #    this = mm.best_overall_layout(
        #    small_qc, self._qpus, successors=True, cost_function=self.accurate_cost_func
        #)
        #else:
        #    this = mm.best_overall_layout(
        #        small_qc, self._qpus, successors=True, cost_function=cost_function, call_limit=500000
        #    )
        #this = self.best_overall_layoutv2(circuit, self._qpus, successors=True, cost_function=cost_function)
        #this = mm.best_overall_layout(small_qc, self._qpus, successors=True, cost_function=cost_function, call_limit=50000)
        #this = self.best_overall_layoutv3(circuit, self._qpus, successors=True, #cost_function=cost_function, call_limit=50000)

        #logger.log(10, "Matched circuit to backend")

        return this
    
    def match_list(self, circuits: List[QuantumCircuit], cost_function=None) -> List:

        #logger = logging.getLogger(__name__)
        #logging.basicConfig(level=10)

        print("Transpiling all circuits")

        try:
            trans_qcs = transpile(circuits, self._qpus[0], optimization_level=3)
        except NameError as e:
            print("Can't transpile on this backend", e)
            return 1
        
        small_qcs = []
        best_layouts = []

        for a, i  in enumerate(trans_qcs):

            print("Matching circuit, num_qubits:{}, depth:{}".format(circuits[a].num_qubits, circuits[a].depth()))
            
            small_qc = mm.deflate_circuit(i)

            best_layouts.append(mm.best_overall_layout(small_qc, self._qpus, successors=True, cost_function=cost_function, call_limit=10000))

        return best_layouts

    def run(self, qernel: Qernel) -> int:

        # Here the matching engine would do its qernel

        logger = logging.getLogger(__name__)
        logging.basicConfig(level=10)

        if qernel.subqernels == []:
            this = self.match(qernel.circuit, cost_function=None)
            qernel.matching = this
        else:
            for i in qernel.subqernels:
                if i.subqernels == []:
                    i.matching = self.match(i.circuit, cost_function=None)
                else:
                    for j in i.subqernels:
                        j.matching = self.match(j.circuit, cost_function=None)

        # Send to multiprogrammer

        #multiprogFifo = open(multiprog_pipe_name, "w")
        #message = str(qernel.id) + "\n"
        #logger.log(10, "Sending qernel to multiprogrammer")
        #multiprogFifo.write(message)
        ## multiprog = Multiprogrammer()
        ## multiprog.submit(qernel)
        #return 0

    def run_list(self, qernels: List[Qernel]) -> int:

        # Running a list is just to make matching faster but it wouldnt be used in a real applicatino of QOS
        # Also, it doesnt support subqernels, it is mainly for evaluating the scheduler

        this = self.match_list([i.circuit for i in qernels], cost_function=None)
        
        for i in range(len(this)):
            qernels[i].matching = this[i]

    # This is to submit a single qernel instead of a whole qernel with subqernels
    def submit_single(self, qernel: Qernel) -> int:

        # Here the matching engine would do its qernel

        qc = QuantumCircuit.from_qasm_str(qernel.circuit)

        # print(self.match(qc, cost_function=self.trivialConstFunction))
        # print("-------------------------")
        # print(self.match(qc, cost_function=None))
        # print("-------------------------")
        self.match(qc, cost_function=self.accurate_cost_func)
        # print("-------------")

        multiprog = Multiprogrammer()
        multiprog.submit(qernel)

        return 0
    
    def results(self):
        pass
