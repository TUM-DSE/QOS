from typing import Any, Dict, List
from qos.types import Engine, Job, QCircuit
from qos.engines.multiprogrammer import Multiprogrammer
import qos.database as db
from qiskit.providers.fake_provider import *
import mapomatic as mm
import redis
from qiskit import transpile, QuantumCircuit
import numpy as np

gates = {
    "u3" : 1,
    "u2" : 1,
    "u1" : 1,
    "cx" : 2,
    "id" : 1,
    "u0" : 1,
    "u" : 1,
    "p" : 1,
    "x" : 1,
    "y" : 1,
    "z" : 1,
    "h" : 1,
    "s" : 1,
    "sdg" : 1,
    "t" : 1,
    "tdg" : 1,
    "rx" : 1,
    "ry" : 1,
    "rz" : 1,
    "sx" : 1,
    "sxdg" : 1,
    "cz" : 1,
    "cy" : 1,
    "swap" : 1,
    "ch" : 1,
    "ccx" : 1,
    "cswap" : 1,
    "crx" : 1,
    "cry" : 1,
    "crz" : 1,
    "cu1" : 1,
    "cp" : 1,
    "cu3" : 1,
    "csx" : 1,
    "cu" : 1,
    "rxx" : 1,
    "rzz" : 1,
    "rccx" : 1,
    "rc3x" : 1,
    "c3x" : 1,
    "c3sqrtx" : 1,
    "c4x" : 1
}


class Matcher(Engine):

    _qpus = []
    _qpu_properties = {}

    def __init__(self) -> None:
        max_qpu_id = 0

        max_qpu_id = db.getLastQPUid()

        for i in range(1, max_qpu_id + 1):
            qpu_name = db.getQPU(i).name
            if "Fake" not in qpu_name:
                continue
            
            backend = eval(qpu_name)()
            self._qpu_properties[backend.name()] = {}
            self._qpu_properties[backend.name()]["medianReadoutError"] = self.getMedianReadoutError(backend)

            basis_gates = backend.configuration().basis_gates
            for g in basis_gates:
                self._qpu_properties[backend.name()][g] = self.getMedianGateError(backend, g)

            self._qpus.append(backend)

    def getMedianReadoutError(self, backend):
        props = backend.properties()
        readouts = []

        for i in range(backend.configuration().n_qubits):
            readouts.append(props.readout_error(i))

        return np.median(readouts)

    
    def getMedianGateError(self, backend, gate):
        props = backend.properties()
        coupling_map = backend.configuration().coupling_map
        qubits = [i for i in range(backend.configuration().n_qubits)]
        errors = []

        if gate == 'reset':
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
            if key == 'measure' or key == 'barrier':
                continue
            for v in range(value):
                fid *= (1 - self._qpu_properties[backend.name()][key])
        
        for i in range(circuit.num_qubits):
            fid *= (1 - self._qpu_properties[backend.name()]["medianReadoutError"])

        error = 1 - fid

        return [(layouts[0], error)]
    
    def accurate_cost_func(self, circ, layouts, backend):
        out = []
        props = backend.properties()
        dt = backend.configuration().dt
        num_qubits = backend.configuration().num_qubits
        t1s = [props.qubit_property(qq, 'T1')[0] for qq in range(num_qubits)]
        t2s = [props.qubit_property(qq, 'T2')[0] for qq in range(num_qubits)]
        for layout in layouts:
            sch_circ = transpile(circ, backend, initial_layout=layout,
                                optimization_level=0, scheduling_method='alap')
            error = 0
            fid = 1
            touched = set()
            for item in sch_circ._data:
                if item[0].name == 'cx':
                    q0 = sch_circ.find_bit(item[1][0]).index
                    q1 = sch_circ.find_bit(item[1][1]).index
                    fid *= (1-props.gate_error('cx', [q0, q1]))
                    touched.add(q0)
                    touched.add(q1)

                elif item[0].name in ['sx', 'x']:
                    q0 = sch_circ.find_bit(item[1][0]).index
                    fid *= 1-props.gate_error(item[0].name, q0)
                    touched.add(q0)

                elif item[0].name == 'measure':
                    q0 = sch_circ.find_bit(item[1][0]).index
                    fid *= 1-props.readout_error(q0)
                    touched.add(q0)

                elif item[0].name == 'delay':
                    q0 = sch_circ.find_bit(item[1][0]).index
                    # Ignore delays that occur before gates
                    # This assumes you are in ground state and errors
                    # do not occur.
                    if q0 in touched:
                        time = item[0].duration * dt
                        fid *= 1-self.idle_error(time, t1s[q0], t2s[q0])

            error = 1-fid
            out.append((layout, error))
            
        return out

    def idle_error(self, time, t1, t2):
        """Compute the approx. idle error from T1 and T2
        Parameters:
            time (float): Delay time in sec
            t1 (float): T1 time in sec
            t2, (float): T2 time in sec
        Returns:
            float: Idle error
        """
        t2 = min(t1, t2)
        rate1 = 1/t1
        rate2 = 1/t2
        p_reset = 1-np.exp(-time*rate1)
        p_z = (1-p_reset)*(1-np.exp(-time*(rate2-rate1)))/2
        return p_z + p_reset
    
    def match(self, circuit : QuantumCircuit, cost_function=None) -> List:
        
        try:
            trans_qc = transpile(circuit, self._qpus[0], optimization_level=3)
        except:
            print("Can't transpile on this backend")

        small_qc = mm.deflate_circuit(trans_qc)

        return mm.best_overall_layout(small_qc, self._qpus, successors=True, cost_function=cost_function)

    def submit(self, job : Job) -> int:

        # Here the matching engine would do its job

        qc = QuantumCircuit.from_qasm_str(job.circuit)

        print(self.match(qc, cost_function=self.trivialConstFunction))
        print("-------------------------")
        print(self.match(qc, cost_function=None))
        print("-------------------------")
        print(self.match(qc, cost_function=self.accurate_cost_func))

        print("-------------")
        multiprog = Multiprogrammer()
        multiprog.submit(job)

        return 0
