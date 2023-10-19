import logging
from time import perf_counter
from dataclasses import dataclass
from multiprocessing import Pool

from qiskit.providers import Job
from qiskit.circuit import QuantumRegister as Fragment
from qiskit.providers.fake_provider import *
from qiskit import transpile

from qvm.qvm.virtual_circuit import VirtualCircuit, generate_instantiations
from qvm.qvm.quasi_distr import QuasiDistr


logger = logging.getLogger("qvm")


@dataclass
class RunTimeInfo:
    run_time: float
    knit_time: float


def run_virtual_circuit(
    virt: VirtualCircuit, shots: int = 20000
) -> tuple[dict[int, float], RunTimeInfo]:
    jobs: dict[Fragment, Job] = {}

    logger.info(
        f"Running virtualizer with {len(virt.fragment_circuits)} "
        + f"{tuple(circ.num_qubits for circ in virt.fragment_circuits.values())} "
        + f"fragments and {len(virt._vgate_instrs)} vgates..."
    )

    num_instances = 0
    now = perf_counter()
    for frag, frag_circuit in virt.fragment_circuits.items():
        instance_labels = virt.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circuit, instance_labels)
        num_instances += len(instantiations)
        backend = FakeKolkataV2()
        comp_instantiations = transpile(instantiations, optimization_level=3)
        #jobs[frag] = virt.get_backend(frag).run(instantiations, shots=shots)
        jobs[frag] = backend.run(comp_instantiations, shots=shots)

    logger.info(f"Running {num_instances} instances...")
    results = {}
    for frag, job in jobs.items():
        counts = job.result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts
        results[frag] = [QuasiDistr.from_counts(c) for c in counts]

    run_time = perf_counter() - now

    logger.info(f"Knitting...")

    with Pool(processes=8) as pool:
        now = perf_counter()
        res_dist = virt.knit(results, pool)
        knit_time = perf_counter() - now

    logger.info(f"Knitted in {knit_time:.2f}s.")

    return res_dist.nearest_probability_distribution(), RunTimeInfo(run_time, knit_time)
