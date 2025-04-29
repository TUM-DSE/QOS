from abc import ABC, abstractmethod
from qos.types.types import Qernel

class ErrorMitigatorPass(ABC):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def run(self, q: Qernel):
        pass


class AnalysisPass(ErrorMitigatorPass):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def run(self, q: Qernel):
        pass

class TransformationPass(ErrorMitigatorPass):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def run(self, q: Qernel):
        pass