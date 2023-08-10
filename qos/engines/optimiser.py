from typing import Any, Dict, List
from qos.engines.matcher import Matcher
from qos.types import Engine, Qernel
import qos.database as db
import pdb
from time import sleep

from qos.tools import debugPrint


class Optimiser(Engine):
    def __init__(self) -> None:
        pass

    def submit(self, qernel: Qernel) -> int:

        # Here the transformer would do its job
        new_qernel = Qernel()
        new_qernel.circuit = qernel.circuit
        new_qernel.provider = qernel.provider
        new_id = db.addQernel(new_qernel)
        new_qernel.id = new_id

        # This would be an example of a case where the transformer didnt modify the qernel and the subqernel would just be the same as the inital qernel
        qernel.subqernels.append(new_qernel.id)
        db.setQernelField(qernel.id, "subqernels", str(qernel.subqernels))

        matcher = Matcher()
        matcher.submit(qernel)

        # print("----------------------------Transformer: ")
        # debugPrint()
        # pdb.set_trace()

        return 0
