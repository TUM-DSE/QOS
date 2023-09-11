from typing import Any, Dict, List
from qos.types import Engine, Qernel
import qos.database as db
import pdb
from time import sleep

from qos.tools import debugPrint


class Virtualizer(Engine):
    def __init__(self) -> None:
        pass

    def submit(self, qernel: Qernel) -> int:

        # Do stuff
        return qernel

        return 0
