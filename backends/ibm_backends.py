from typing import Any, Dict, Optional
from warnings import warn

import mapomatic as mm
import qiskit.providers.fake_provider as FakeAccountProvider
from qiskit.compiler import transpile
from qiskit.providers.ibmq import AccountProvider
from qiskit.providers.ibmq.ibmqbackend import IBMQSimulator
from qiskit.providers.models.backendproperties import BackendProperties


class IBMQPU:
    def __init__(
        self, backend_name: str, provider: Optional[AccountProvider] = None
    ) -> None:
        if isinstance(provider, AccountProvider):
            backend = provider.get_backend(backend_name)

            if isinstance(backend, IBMQSimulator):
                raise ValueError(
                    "Simulators are not currently supported. Please choose a quantum backend."
                )
        elif "Fake" in backend_name:
            if provider is not None:
                warn("AccountProvider passed but fake backend requested.")
            backend = getattr(FakeAccountProvider, backend_name)()

            # Need to do some modifications for compatility with mapomatic if V2
            if "V2" in backend_name:
                v1_backend = getattr(
                    FakeAccountProvider, backend_name.replace("V2", "")
                )()

                setattr(
                    backend,
                    "_properties",
                    BackendProperties.from_dict(v1_backend.properties().to_dict()),
                )

                def properties(self):
                    return self._properties

                backend.properties = MethodType(properties, backend)
        else:
            raise RuntimeError(
                "Either an AccountProvider or a name of a fake backend must be provided. "
                "Names of fake backends should be of format `Fake{NameOfBackend}` with `V2` "
                "appended to the name if needed. Examples: FakeLondon, FakeAthensV2."
            )

        self._backend = backend
        # self._qernels: Dict[int, Qernel] = {}
        self._qid_ctr: int = 0

    # TODO - Is this method still needed?
    # def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
    #   self._qernels[self._qid_ctr] = qernel
    #  self._qid_ctr += 1
    # return self._qid_ctr - 1

    # def execute_qernel(self, qernel: Qernel, args: QernelArgs, shots: int) -> None:
    # circ = qernel.with_input(args=args)

    def cost(self, qernel: Qernel) -> float:
        trans_qc = transpile(
            circuits=qernel, backend=self._backend, optimization_level=3
        )
        small_qc = mm.deflate_circuit(input_circ=trans_qc)
        layouts = mm.matching_layouts(circ=small_qc, cmap=self._backend)
        scores = mm.evaluate_layouts(
            circ=small_qc, layouts=layouts, backend=self._backend
        )

        return scores[0][1]

    def overhead(self, qid: int) -> int:
        pass
