from qos.kernel.matcher import Matcher
from qos.kernel.multiprogrammer import Multiprogrammer
from qos.kernel.scheduler import Scheduler
import os
from qiskit.circuit.random import random_circuit
from benchmarks.circuits import get_circuits, BENCHMARK_CIRCUITS
from qos.types import Qernel
import logging
from qiskit.providers import fake_provider
from qiskit_ibm_provider import IBMProvider
from qiskit.compiler import transpile
import pdb
import qos.database as db
import matplotlib.pyplot as plt
from qos.time_estimator.basic_estimator import CircuitEstimator
from qos.secrets import IBM_TOKEN

#How many circuit submission per second
SUB_FREQ = 5
CIRC_NUMBER = 16
CIRC_SIZE = 8
CIRC_DEPTH = 5

FID_WEIGHTS = [0.7,0.8,0.9,1]

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
      print("Matching circuit {}...".format(str(i)))
      new_qernel = Qernel(random_circuit(CIRC_SIZE, CIRC_DEPTH, max_operands=2, measure=True))
      #submit time in seconds times the sub_freq
      new_qernel.submit_time = i*SUB_FREQ
      new_qernel.id = str(i)  
      matcher.run(new_qernel)
      qernels.append(new_qernel)
    
    sched = Scheduler()

    print("Working on scheduler...")

    avg_error_list = []
    avg_wait_list = []

    for fid in FID_WEIGHTS:
      qernels_copy = qernels.copy()

      db.reset_local_queues()
      all_queues = sched.run(qernels_copy, "balanced", fid, 0)

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
            avg_error += (1-j[4])
            avg_wait += j[3]

      #pdb.set_trace()

      avg_error_list.append(avg_error/CIRC_NUMBER)
      avg_wait_list.append(avg_wait/CIRC_NUMBER)

      #plt.figure(figsize=(max([i[1] for i in dist]),len(dist)+3))

      plt.bar(range(len(dist)), [i[1] for i in dist], align='center')
      #fig, ax = plt.subplots()
      plt.xticks(range(len(dist)), [i[0] for i in dist])
      plt.xticks(rotation=30)
      plt.text(len(dist), max([i[1] for i in dist])-2, 'Avg fidelity:{}\nAvg waiting time:{}s\n'.format(round(avg_error/CIRC_NUMBER,3), round((avg_wait/CIRC_NUMBER),3)), fontsize=12, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))
      #ax.text(1.03, 0.98, my_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
      #plt.autoscale(enable=True, axis='x', tight=True)
      #fig.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
      plt.tight_layout()
      plt.savefig('dist{}.png'.format(str(fid)))
      plt.close()
      #plt.show()
      #all_queues = sched.run(qernels, "bestqpu")

    #pdb.set_trace()

    fig, ax1 = plt.subplots()

    # Plot avg fidelity on the left y-axis
    ax1.plot(FID_WEIGHTS, avg_error_list, label='Avg fidelity', color='b')
    ax1.set_xlabel('Fidelity weight')
    ax1.set_ylabel('Avg fidelity', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')

    # Create a secondary y-axis on the right
    ax2 = ax1.twinx()

    # Plot avg waiting time on the right y-axis
    ax2.plot(FID_WEIGHTS, avg_wait_list, label='Avg waiting time', color='r')
    ax2.set_ylabel('Avg waiting time', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')

    plt.title('Avg Fidelity and Waiting Time')
    plt.savefig('avg_error_and_wait.png')
    plt.show()
    
    #Plot two lines: avg_error_list and avg_wait_list vs FID_WEIGHTS
    #plt.plot(FID_WEIGHTS, avg_error_list, label='Avg fidelity')
    #plt.plot(FID_WEIGHTS, avg_wait_list, label='Avg waiting time')
    #plt.xlabel('Fidelity weight')
    #plt.ylabel('Avg fidelity and waiting time')
    #plt.legend()
    #plt.savefig('avg_error_and_wait.png')
    #plt.close()
   

      #plot dist as a bar chart and save to file
main()

#all_circuits = gen_all_circits([10, 11])

#print(all_circuits[0])

#pdb.set_trace()

#new_qernel = Qernel(all_circuits[0])

#estimator(new_qernel)