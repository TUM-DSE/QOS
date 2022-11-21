from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple, Union

from qiskit.circuit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit

from qstack.qvm.virtual_gate import VirtualBinaryGate


@dataclass(frozen=True)
class UserConfiguration:
    """
    This class contains the expected parameters from a user when submitting
    a circuit to the QStack. The currently implemented parameters are as follows:

    - circuit: This is the main circuit that the user wishes to execute on some backend.
        This is the only required attribute for a user's submission.
    - virtualization_technique: the user's choice of virtualization methodology. One can
        either decide to not virtualize any gate in the circuit at all (the option "None"),
        or choose exact virtualization, incurring 6^k overhead (the option "Default"), or
        choose a middle ground of approximate virtualization, incurring only 2^k overhead
        (the option "Approximate").
    - bisection_policy: the policy upon which the choice of gates to virtualize is made.
        The user can choose either Kernighan-Lin or Ladder bisection, or provide their own
        custom callable function to carry out the bisection.
    - custom_virtualizations_dict: a dictionary containing the key-value mapping of the user's
        custom virtualized gates. This can be used to override existing classes or define
        virtualizations for new gates.
    - maximum_virtualizations: the maximum number of the virtualizations the QVM is
        allowed to make. The default is to virtualize all multi-qubit gates.
    - hardware_constraints: a dictionary where keys are hardware properties and the values
        are tuples containing the user-imposed constraint on the respective hardware property,
        and a boolean flag as to whether this constrain is hard.
    - allow_run_on_simulators: a boolean flag indicating whether the user is allowing the circuit
        or its fragments to run on simulators.

    """

    circuit: QuantumCircuit

    virtualization_technique: Literal["None", "Default", "Approximate"] = "None"
    bisection_policy: Union[
        Literal["kernighan-lin", "ladder"], Callable[[DAGCircuit], DAGCircuit]
    ] = "kernighan-lin"
    custom_virtualizations_dict: Optional[Dict[str, VirtualBinaryGate]] = None
    maximum_virtualization_count: Optional[int] = None

    hardware_constraints: Optional[
        Dict[
            Literal["T1", "T2", "CNOTError", "ReadoutError", "MapomaticError"],
            Tuple[float, bool],
        ]
    ] = None
    allow_run_on_simulators: bool = False

    @staticmethod
    def from_dictionary(input_dict: dict) -> "UserConfiguration":
        return UserConfiguration(**input_dict)
