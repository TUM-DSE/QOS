from types import MethodType
from typing import Any, Dict, Optional
from qos.backends.types import QPU
from qiskit_ibm_provider import IBMProvider
from qiskit import compiler
import logging
from qiskit.providers import fake_provider
import pdb

# from qos.scheduler_old import Scheduler, scheduler_policy, fifo_policy
# from qstack.qernel import Qernel, QernelArgs
# from qstack.types import QPUWrapper


class IBMQPU(QPU):
    logger: logging.Logger

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.provider = IBMProvider.load_account()
        return

    def run(self, circuit, backend, nshots) -> int:
        self.logger.log(10, "Running circuit")
        backend_obj = getattr(fake_provider, backend)()
        qernel = backend_obj.run(circuit, shots=int(nshots))
        return qernel.result()

    def transpile(self, circuit, backend) -> int:
        self.logger.log(10, "Transpiling qernel")
        backend_obj = getattr(fake_provider, backend)()
        # pdb.set_trace()
        # backend = [backend for backend in backends() if backend.name() == backend][0
        # backend = FakeProviderForBackendV2().get_backend(backend)
        qc = compiler.transpile(circuit, backend_obj)
        return qc
