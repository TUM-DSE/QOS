import logging
from multiprocessing.dummy import Pool
from time import time
from typing import Dict


from vqc.circuit import DistributedCircuit
from .frag_executor import FragmentExecutor
from .knit import knit


def execute(vc: DistributedCircuit, shots: int = 10000) -> Dict[str, int]:
    frag_execs = [FragmentExecutor(vc, fragment) for fragment in vc.fragments]
    exec_time = time()

    with Pool(processes=len(frag_execs)) as pool:
        pool.map(lambda x: x.execute(shots), frag_execs)

    logging.info(f"Execution time: {time() - exec_time}")

    knit_time = time()
    res = knit(frag_execs, vc.virtual_gates)
    logging.info(f"Knit time: {time() - knit_time}")
    return res.counts(shots)
