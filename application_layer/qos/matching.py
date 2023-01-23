from typing import Any, Dict, List
import json
import os
from qstack.qernel.qernel import Qernel, QernelArgs
from qstack.types import QOSEngineI, scheduler_policy, Job, Scheduler_base


class Qernel_DB():

    path:str

    def __init__(self, path):

        #self.db_path = os.path.expanduser(path) #Need to implement here some error capturing procedure try except maybe
        pass

    def open():
        pass

    def load(self, data:Dict):
        pass

    def set(self, qid:int, key:str, value:float):
        pass

    def get(self, qid:int, key:str):
        pass

    def dumpDB(self):
        pass

    def delete(self, qid:int):
        pass



class matching_engine(QOSEngineI):

    registry:Qernel_DB

    def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
        pass

    def execute_qernel(self, qid: int, args: QernelArgs, shots: int) -> None:
        pass
