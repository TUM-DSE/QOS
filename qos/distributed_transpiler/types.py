from abc import ABC, abstractmethod
from qos.types import Qernel

class DistributedTranspilerPass(ABC):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def run(self, q: Qernel):
        pass


class AnalysisPass(DistributedTranspilerPass):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def run(self, q: Qernel):
        pass

class TransformationPass(DistributedTranspilerPass):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def run(self, q: Qernel):
        pass