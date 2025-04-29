import benchmarks.circuits.circuits as bench
from qos.kernel.multiprogrammer import Multiprogrammer
from qos.types.types import Qernel
import pdb
from qos.error_mitigator.analyser import BasicAnalysisPass, SupermarqFeaturesAnalysisPass

backends = [
    'FakeCairo',
    'FakeWashington',
    'FakeMumbai',
    'FakeMontreal',
    'FakeKolkata',
    'FakeHanoi',
    'FakeCambridge',
    'FakeSydney',
    'FakeToronto',
    'FakeRochester',
    'FakeBrooklyn',
]

def gen_all_benchmarks_wsize(size):

    analyser1 = BasicAnalysisPass()
    analyser2 = SupermarqFeaturesAnalysisPass()

    for benchmark in bench.BENCHMARK_CIRCUITS:
        qernels = []
        try:
            circ = bench.get_circuits(benchmark,[size,size+1])
        except:
            continue

        if circ is not None:
            new_qernel = Qernel(*circ)
            analyser1.run(new_qernel)
            analyser2.run(new_qernel)
            qernels.append(new_qernel)
        
        print(circ)
    return qernels


def main():
    circs1 = gen_all_benchmarks_wsize(10)
    circs2 = gen_all_benchmarks_wsize(10)
    
    #pdb.set_trace()

    multi = Multiprogrammer()
    max_score = (0,0,0)

    for b1 in circs1:
        for b2 in circs2:
                tmp = multi.get_matching_score(b1,b2)
                if tmp > max_score[0]:
                    max_score = (tmp,b1,b2)

    print(max_score)

main()