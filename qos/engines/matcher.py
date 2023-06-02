from typing import Any, Dict, List
from qos.types import Engine, Job, QCircuit
from qos.engines.multiprogrammer import Multiprogrammer
import qos.database as db


class Matcher(Engine):
    def submit(self, job: Job) -> int:

        # Here the matching engine would do its job

        multiprog = Multiprogrammer()
        multiprog.submit(job)

        return 0
