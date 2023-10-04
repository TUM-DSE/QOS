from qos.kernel.matcher import Matcher
from qos.kernel.multiprogrammer import Multiprogrammer
from qos.kernel.scheduler import Scheduler
import os
from qiskit.circuit.random import random_circuit
from qos.types import Qernel
import logging
import pdb
import matplotlib.pyplot as plt

#How many circuit submission per second
SUB_FREQ = 0.01
CIRC_NUMBER = 32
CIRC_SIZE = 7
CIRC_DEPTH = 5

def main():
    
    try:
        os.system(f'redis-cli flushall')   
        os.system(f'python load_qpus.py')
    except FileNotFoundError:
      print(f"Error: The file does not exist.")

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=50)

    matcher = Matcher()
    qernels = []

    for i in range(CIRC_NUMBER):
      new_qernel = Qernel(random_circuit(CIRC_SIZE, CIRC_DEPTH, max_operands=2, measure=True))
      #submit time in nanoseconds times the sub_freq
      new_qernel.submit_time = i*SUB_FREQ*1000000000
      new_qernel.id = str(i)  
      matcher.run(new_qernel)
      qernels.append(new_qernel)
    
    sched = Scheduler()

    all_queues = sched.run(qernels, "balanced")
    #all_queues = sched.run(qernels, "bestqpu")

    #pdb.set_trace()

    #plot dist as a bar chart and save to file
    avg_error = 0
    avg_wait = 0

    #pdb.set_trace()

    dist = []

    for i in all_queues:
      dist.append([i[0], len(i[1])])
      if i[1] == []:
        continue
      else:
        for j in i[1]:
          avg_error += j[4]
          avg_wait += j[3]

    avg_error = avg_error/CIRC_NUMBER
    avg_wait = avg_wait/CIRC_NUMBER

    #plt.figure(figsize=(max([i[1] for i in dist]),len(dist)+3))

    plt.bar(range(len(dist)), [i[1] for i in dist], align='center')
    plt.text(len(dist), max([i[1] for i in dist])-2, 'Avg fidelity:{}\nAvg waiting time:{}s\n'.format(round(1-avg_error,3), round(avg_wait/1000000000,3)), fontsize=12, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))
    #ax.text(1.03, 0.98, my_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    plt.xticks(range(len(dist)), [i[0] for i in dist])
    plt.tight_layout()
    plt.savefig('dist.png')
    plt.show()

main()