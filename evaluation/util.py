from evaluation.evaluation_config import *
from benchmarks.circuits import get_circuit, BENCHMARK_CIRCUITS
from qos.types.types import Qernel


RANDOM_TIME = lambda : np.clip(np.random.normal(TIME_MEAN, TIME_DEVIATION),a_min=0, a_max=None)
RANDOM_WIDTH = lambda: randint(WIDTH_MIN, WIDTH_MAX)
RANDOM_SHOTS = lambda : np.clip(np.random.normal(SHOTS_MEAN, SHOTS_DEVIATION),a_min=0, a_max=None)

def random_width():
  tmp = RANDOM_WIDTH()
  while (tmp//2)%2!=0:
    tmp = RANDOM_WIDTH()

  return int(tmp)

def random_shots():
  tmp = RANDOM_SHOTS()
  while tmp==0:
    tmp = RANDOM_SHOTS()

  return int(tmp)

def random_time():
  tmp = RANDOM_TIME()
  while tmp==0:
    tmp = RANDOM_TIME()

  return tmp

def get_random_circuit(next_qernel_id, current_time):
    #Since I am using 30x factor, each random bechmark will be halved in terms of sized copied 30 times
    random_bench = BENCHMARK_CIRCUITS[randint(0, len(BENCHMARK_CIRCUITS)-1)]
    #pdb.set_trace()
    width = random_width()//2
    
    all_cuts = []
    for i in range(CIRC_CUT_FACTOR):
        new_qernel = Qernel(get_circuit(random_bench, width))
        new_qernel.submit_time = current_time
        new_qernel.id = str(next_qernel_id)
        next_qernel_id += 1
        all_cuts.append(new_qernel)

    return all_cuts, next_qernel_id