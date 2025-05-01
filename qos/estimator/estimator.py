from typing import List
import logging
from random import choice
import pickle
import joblib

from qos.types.types import Engine, Qernel
import qos.database as db
from data.ibm_token import IBM_TOKEN

from qiskit_ibm_runtime.fake_provider import *
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime.models import BackendProperties
from qiskit import transpile, QuantumCircuit

import mapomatic as mm
import mapomatic.layouts as mply
import numpy as np

logger = logging.getLogger('qiskit')
logger.setLevel(logging.ERROR)
logger = logging.getLogger('stevedore')
logger.setLevel(logging.ERROR)

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


class Estimator(Engine):

    _qpus = []
    _qpu_properties = {}
    

    def __init__(self, qpus: list = None, model_path: str = "qos/estimator") -> None:
        max_qpu_id = 0

        max_qpu_id = db.getLastQPUid()
        self.model = joblib.load(model_path)  # scikit-learn model

        if qpus == None:
            for i in range(1, max_qpu_id + 1):
                qpu_name = db.getQPU(i).name
                qpu_alias = db.getQPU(i).alias
        
                if "Fake" in qpu_name:
                    backend = eval(qpu_name)()
                    self._qpu_properties[backend.name] = {}
                    self._qpu_properties[backend.name][
                    "medianReadoutError"] = self.getMedianReadoutError(backend)
                    db.setQPUField(i, "backend", pickle.dumps(backend))
                    basis_gates = backend.configuration().basis_gates
                    for g in basis_gates:
                        self._qpu_properties[backend.name][g] = self.getMedianGateError(backend, g)
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
                self._qpu_properties[q.name] = {}
                self._qpu_properties[q.name]["medianReadoutError"] = self.getMedianReadoutError(q)

                non_local_gate_error = []
                basis_gates = q.configuration().basis_gates                
                for g in basis_gates:
                    self._qpu_properties[q.name][g] = self.getMedianGateError(q, g)
                    if gates[g] == 2:
                        non_local_gate_error .append(self._qpu_properties[q.name][g])

                self._qpu_properties[q.name]["medianNonLocalError"] = np.median(non_local_gate_error)
                self._qpu_properties[q.name]["medianT1"] = self.getMedianT1(q)
                self._qpu_properties[q.name]["medianT2"] = self.getMedianT2(q)

            best_readout = 1
            best_T2 = 0
            best_readout_machine = ""
            best_T2_machine = ""
            best_nonlocal_error = 1
            best_nonlocal_machine = ""

            best_overall_score = 0
            best_overall_machine = ""

            for k,v in self._qpu_properties.items():
                medianReadout = v["medianReadoutError"]
                medianT2 = v["medianT2"]
                medianNonLocal = v["medianNonLocalError"]
                score = ((1-medianReadout) + (1-medianNonLocal) + (medianT2 / 200)) / 3

                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_machine = k

                if medianReadout < best_readout:
                    best_readout = medianReadout
                    best_readout_machine = k
                if medianT2 > best_T2:
                    best_T2 = medianT2
                    best_T2_machine = k
                if medianNonLocal < best_nonlocal_error:
                    best_nonlocal_error =medianNonLocal
                    best_nonlocal_machine = k

            print(best_nonlocal_machine, best_readout_machine, best_T2_machine)
            print(best_overall_machine)

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
        props:BackendProperties = backend.properties()
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
                try:
                    errors.append(props.gate_error(gate, pair))
                except:
                    # If the qubit pair is not available for on the simplified pair list for this gate, we take a random one
                    errors.append(props.gate_error(gate, choice(list(props._gates[gate].keys()))))

        return np.median(errors)
    

    def getMedianT1(self, backend):
        props = backend.properties()
        num_qubits = backend.configuration().num_qubits

        t1s = []
        average_t1 = 0

        for qq in range(num_qubits):
           
            try:
                t1 = props.qubit_property(qq, "T1")[0]
                t1s.append(t1)
                average_t1 = average_t1 + t1
            except:
                t1s.append(average_t1 / len(t1s))

        return np.median(t1s)

    def getMedianT2(self, backend):
        props = backend.properties()
        num_qubits = backend.configuration().num_qubits

        t2s = []
        average_t2 = 0

        for qq in range(num_qubits):
            try:
                t2 = props.qubit_property(qq, "T2")[0]
                t2s.append(t2)
                average_t2 = average_t2 + t2
            except:
                t2s.append(average_t2 / len(t2s))

        return np.median(t2s)

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

        t1s = []
        t2s = []
        average_t2 = 0

        for qq in range(num_qubits):
            t1s.append(props.qubit_property(qq, "T1")[0])
            try:
                t2 = props.qubit_property(qq, "T2")[0]
                t2s.append(t2)
                average_t2 = average_t2 + t2
            except:
                t2s.append(average_t2 / len(t2s))


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

    def regression_cost_function(self, qernel: Qernel) -> float:
        circuit = qernel.circuit
        features = self._extract_features(circuit)
        fidelity = self.model.predict([features])[0]
        return float(fidelity)

    def idle_error(self, time, t1, t2):
        t2 = min(t1, t2)
        rate1 = 1 / t1
        rate2 = 1 / t2
        p_reset = 1 - np.exp(-time * rate1)
        p_z = (1 - p_reset) * (1 - np.exp(-time * (rate2 - rate1))) / 2
        return p_z + p_reset
    
    def best_overall_layoutv2(self, circuit, backends, successors=True, cost_function=trivialConstFunction):
        """
        Determines the best overall layout for a quantum circuit across multiple backends.

        This function evaluates the given quantum circuit on one or more backends, 
        identifies the best transpiled circuit based on the number of non-local gates, 
        and computes the optimal layout for the circuit on each backend. The results 
        are sorted by the specified cost function.

        Args:
            circuit (QuantumCircuit): The quantum circuit to be evaluated.
            backends (List[Backend]]): A list of backends 
                to evaluate the circuit on.
            successors (bool, optional): If True, returns a list of all layouts sorted by 
                the cost function. If False, returns only the best layout. Defaults to True.
            cost_function (callable, optional): A custom cost function to evaluate layouts. 
                If None, a default cost function is used. Defaults to trivialConstFunction.

        Returns:
            Union[List[Tuple[Layout, str, float]], Tuple[Layout, str, float], List]: 
                - If `successors` is True, returns a list of tuples, where each tuple contains:
                    - Layout: The optimal layout for the circuit.
                    - str: The name of the backend.
                    - float: The cost associated with the layout.
                - If `successors` is False, returns the best layout tuple.
                - If no valid layouts are found, returns an empty list.

        Raises:
            NameError: If the circuit cannot be transpiled on a given backend.

        Notes:
            - The function assumes that the circuit gates are compatible with the backend's 
              basis gates. If not, the backend is skipped.
            - Only backends that are not simulators and have sufficient qubits to support 
              the circuit are considered.
            - The function uses a matching algorithm to find candidate layouts and evaluates 
              them using the provided or default cost function.
        """
        if backends is None:
            backends = self._qpus

        if not isinstance(backends, list):
            backends = [backends]

        if cost_function is None:
            cost_function = mply.default_cost

        best_out = []

        for backend in backends:
            config = backend.configuration()

            try:
                trans_qc_list = transpile([circuit]*20, backend, optimization_level=3)
                best_cx_count = [circ.num_nonlocal_gates() for circ in trans_qc_list]
                best_idx = np.argmin(best_cx_count)
                trans_qc = trans_qc_list[best_idx]

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
                layouts = mply.matching_layouts(circ, config.coupling_map, call_limit=int(1e3))
                layout_and_error = mply.evaluate_layouts(circ, layouts, backend,
                                                    cost_function=cost_function)
                if any(layout_and_error):
                    for l in layout_and_error:
                        if len(l[0]) == circuit.num_qubits:
                            layout = l[0]
                            error = l[1]
                            best_out.append((layout, config.backend_name, error))
                            break                    
        best_out.sort(key=lambda x: x[2])
        if successors:
            return best_out
        if best_out:
            return best_out[0]
        return best_out
    

    def best_overall_layoutv3(self, circuit, backends, successors=True, cost_function=trivialConstFunction):
        if backends is None:
            backends = self._qpus
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

    def run(self, qernel: Qernel, successors=True, cost_function=trivialConstFunction):
        return self.best_overall_layoutv2(qernel.circuit, successors=successors, cost_function=cost_function)
