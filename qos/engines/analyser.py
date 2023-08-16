from typing import Any, Dict, List
from qos.types import Engine, Qernel
from qos.engines.multiprogrammer import Multiprogrammer
import qos.database as db
from qiskit.providers.fake_provider import *
import mapomatic as mm
import logging
import pdb
import redis
from qiskit import transpile, QuantumCircuit
import numpy as np


class Analyser(Engine):

    def __init__(self) -> None:
        pass

    @staticmethod
    def analyse(qernel: Qernel) -> Qernel:
        # TODO - This is where the analysis would happen
        return qernel