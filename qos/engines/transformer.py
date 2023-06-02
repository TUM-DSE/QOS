from typing import Any, Dict, List
from qos.engines.matcher import Matcher
from qos.types import Engine, Job, QCircuit
import qos.database as db


class Transformer(Engine):
    def __init__(self) -> None:
        pass

    def submit(self, circuits: List[QCircuit]) -> int:

        # Here the transformer would do its job

        matcher = Matcher()
        matcher.submit(job)

        return 0
