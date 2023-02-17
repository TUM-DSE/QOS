from .chemistry import HamiltonianSimulationBenchmark, VQEBenchmark
from .error_correction import BitCodeBenchmark, PhaseCodeBenchmark
from .optimization import VanillaQAOABenchmark, FermionicSwapQAOABenchmark
from .quantum_information import GHZBenchmark
from ._types import Device
from ._utils import perfect_counts, fidelity

# from .sim import SimDevice
# from .fidelity import fidelity
