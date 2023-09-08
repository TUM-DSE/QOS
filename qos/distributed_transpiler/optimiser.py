from typing import Any, Dict, List
from qos.engines.virtualizer import Virtualizer
from qos.types import Engine, Qernel
import qos.database as db
import pdb
from time import sleep

from qos.tools import debugPrint


class Optimiser(Engine):
    def __init__(self) -> None:
        pass

    def run(self, qernel: Qernel) -> int:

        # Here the Optimizer would do its job, but for now we just copy the qernel and adds it as a subqernel to the original
        new_qernel = Qernel()
        new_qernel.circuit = qernel.circuit
        new_qernel.provider = qernel.provider
        new_id = db.addQernel(new_qernel)
        new_qernel.id = new_id

        # This is the list of transformations that have been applied to the qernel in the order they were applied
        qernel.transformations = []
        db.setQernelField(qernel.id, "transformations", str(qernel.transformations))

        # This would be an example of a case where the transformer didnt modify the qernel and the subqernel would just be the same as the inital qernel
        qernel.subqernels.append(new_qernel)
        db.addSubqernel(qernel.id, new_qernel.id)

        #Virtualizer = Virtualizer()
        #Virtualizer.submit(qernel)

        # print("----------------------------Transformer: ")
        # debugPrint()
        # pdb.set_trace()

        return qernel
    
    def results(self) -> None:
        pass
