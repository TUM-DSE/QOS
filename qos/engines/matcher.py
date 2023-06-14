from typing import Any, Dict, List
from qos.types import Engine, Job, QCircuit
from qos.engines.multiprogrammer import Multiprogrammer
import qos.database as db
from qiskit.providers.fake_provider import *
import mapomatic as mm
import redis
from qiskit import transpile, QuantumCircuit

class Matcher(Engine):

    _qpus = []

    def __init__(self) -> None:
        max_qpu_id = 0

        max_qpu_id = db.getLastQPUid()

        for i in range(1, max_qpu_id + 1):
            qpu_name = db.getQPU(i).name
            if "Fake" not in qpu_name:
                continue
            
            backend = eval(qpu_name)()
            self._qpus.append(backend)

    def mapomatic_default(self, circuit : QuantumCircuit) -> List:
        
        try:
            trans_qc = transpile(circuit, self._qpus[0], optimization_level=3)
        except:
            print("Can't transpile on this backend")

        small_qc = mm.deflate_circuit(trans_qc)

        return mm.best_overall_layout(small_qc, self._qpus, successors=True)

    def submit(self, job : Job) -> int:

        # Here the matching engine would do its job

        qc = QuantumCircuit.from_qasm_str(job.circuit)

        print(self.mapomatic_default(qc))

        print("-------------")
        multiprog = Multiprogrammer()
        multiprog.submit(job)

        return 0
