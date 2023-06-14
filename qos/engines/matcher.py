from typing import Any, Dict, List
from qos.types import Engine, Job, QCircuit
from qos.engines.multiprogrammer import Multiprogrammer
import qos.database as db
from qiskit.providers.fake_provider import *
import mapomatic as mm
from qiskit import transpile, QuantumCircuit

class Matcher(Engine):

    _qpus = List[FakeBackendV2]

    def __init__(self) -> None:
        max_qpu_id = 0

        with redis.Redis() as db:
            max_qpu_id = db.get("qpuCounter")
        
        for i in range(max_qpu_id):
            _qpus.append(db.getJobField(i, "backend")

    def mapomatic_default(circuit : QuantumCircuit) -> List:
        
        try:
            trans_qc = transpile(circuit, _qpus[0], optimization_level=3)
        except:
            print("Can't transpile on this backend")

        small_qc = mm.deflate_circuit(trans_qc)

        return mm.best_overall_layout(small_qc, backends, successors=True)

    def submit(self, job : Job) -> int:

        # Here the matching engine would do its job

        print(mapomatic_default(QuantumCircuit.from_qasm_str(job.circuit)))

        print("-------------")
        multiprog = Multiprogrammer()
        multiprog.submit(job)

        return 0
