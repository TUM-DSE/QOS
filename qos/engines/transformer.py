from typing import Any, Dict, List
from qos.engines.matcher import Matcher
from qos.types import Engine, Job, QCircuit
import qos.database as db
import pdb


class Transformer(Engine):
    def __init__(self) -> None:
        pass

    def submit(self, job: Job) -> int:

        # Here the transformer would do its job
        new_subjob = Job()
        new_subjob.args["circuit"] = job.circuit
        new_id = db.addJob(new_subjob)
        new_subjob.id = new_id

        # This would be an example of a case where the transformer didnt modify the job and the subjob would just be the same as the inital job
        job.subjobs.append(new_subjob.id)
        db.setJobField(job.id, "subjobs", str(job.subjobs))

        matcher = Matcher()
        # pdb.set_trace()
        matcher.submit(job)

        return 0
