import itertools
from multiprocessing import Pool, cpu_count
from typing import Iterable, Iterator, List, Tuple
from vqc.executor.frag_executor import FragmentExecutor
from vqc.prob import ProbDistribution
from vqc.virtual_gate.virtual_gate import VirtualBinaryGate


def chunk(l: Iterable, n: int) -> Iterator[List]:
    it = iter(l)
    while True:
        ch = list(itertools.islice(it, n))
        if not ch:
            return
        yield ch


def merge(prob_dists: List[ProbDistribution]) -> ProbDistribution:
    assert len(prob_dists) > 0
    res = prob_dists[0]
    for prob_dist in prob_dists[1:]:
        res = res.merge(prob_dist)
    return res


def merge_iter(
    frag_execs: List[FragmentExecutor], config_ids: Iterator[Tuple[int, ...]]
) -> Iterator[List[ProbDistribution]]:
    for config_id in config_ids:
        yield [fe.get_result(config_id) for fe in frag_execs]


def knit(
    frag_execs: List[FragmentExecutor], vgates: List[VirtualBinaryGate]
) -> ProbDistribution:
    conf_l = [list(range(len(vgate.configure()))) for vgate in vgates]
    config_ids = iter(itertools.product(*conf_l))

    with Pool(processes=cpu_count()) as pool:
        miter = list(merge_iter(frag_execs, config_ids))
        results = pool.map(merge, miter)

        while len(vgates) > 0:
            vgate = vgates.pop(-1)
            chunks = list(chunk(list(results), len(vgate.configure())))
            results = pool.map(vgate.knit, chunks)
    return results[0]
