from qos.kernel.matcher import Matcher
from qos.kernel.multiprogrammer import Multiprogrammer
from qos.kernel.scheduler import Scheduler
import os
from qiskit.circuit.random import random_circuit
from benchmarks.circuits import get_circuit, BENCHMARK_CIRCUITS
from benchmarks.plot.util import plot_lines_2yaxis, bar_plot, save_figure
from qos.types import Qernel
import logging
from qiskit.providers import fake_provider
from qiskit_ibm_provider import IBMProvider
from qiskit.compiler import transpile
import pdb
import qos.database as db
import matplotlib.pyplot as plt
from qos.time_estimator.basic_estimator import CircuitEstimator
from ibm_token import IBM_TOKEN
from random import randint
import pandas as pd
import seaborn as sns
import numpy as np

#This number is just the number of uncut circuits, this will be multiplied by CIRC_CUT_FACTOR
CIRC_NUMBER = 5
CIRC_CUT_FACTOR = 20 #This means that each random benchmark will be halved in terms of size and copied 30 times
NCLIENTS = 7
#Circuit width is defined by an uniform distribution
WIDTH_MIN = 8
WIDTH_MAX = 24

#Circuit shots is defined by an uniform distribution
SHOTS_DEVIATION = 2048
SHOTS_MEAN = 8192

#Submittion time interval is defined by a guassion function
TIME_DEVIATION = (NCLIENTS/7)*0.22 #This might not be correct but should be close
TIME_MEAN = (NCLIENTS/7)*3.5

#TIME_DEVIATION = 0.25
#TIME_MEAN = 3.5

#Circuit depth is defined by a guassion function 
#DEPTH_DEVIATION = 50
#DEPTH_MEAN = 0

RANDOM_TIME = lambda : np.clip(np.random.normal(TIME_MEAN, TIME_DEVIATION),a_min=0, a_max=None)
RANDOM_WIDTH = lambda: randint(WIDTH_MIN, WIDTH_MAX)
RANDOM_SHOTS = lambda : np.clip(np.random.normal(SHOTS_MEAN, SHOTS_DEVIATION),a_min=0, a_max=None)

#FID_WEIGHTS = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
FID_WEIGHTS = [0.6,0.7,0.8,0.9,1]

#RANDOM_DEPTH = lambda : np.clip(np.random.normal(DEPTH_MEAN, DEPTH_DEVIATION),a_min=0, a_max=None)

# The benchmark only work with even number of qubits and since we will be working with half the size of the this random number we need to make sure that half of it is even
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

#def random_depth():
#  tmp = RANDOM_DEPTH()
#  while tmp==0 or tmp>200:
#    tmp = RANDOM_DEPTH()
#
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


def hard_job():
    try:
        os.system(f'redis-cli flushall')   
        os.system(f'python load_qpus.py')
    except FileNotFoundError:
        print(f"Error: The file does not exist.")

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=50)
    matcher = Matcher()
    qernels = []

    current_time = 0
    next_qernel_id = 0

    for i in range(CIRC_NUMBER):
        current_time += random_time()
      
        #This is will generate 30 copies of a random benchmark divided with half the qubits
        new_qernels, next_qernel_id = get_random_circuit(next_qernel_id, current_time)      
      
        qernels += new_qernels

    matcher.run_list(qernels)
    #pdb.set_trace()
    sched = Scheduler()

    print("Working on scheduler...")

    all_all_queues = []

    avg_fid_list = []
    avg_wait_list = []

    for fid in FID_WEIGHTS:
        qernels_copy = qernels.copy()

        db.reset_local_queues()
        all_queues = sched.run(qernels_copy, "balanced", fid, 0)
        all_all_queues.append(all_queues)

        avg_fid = 0
        avg_wait = 0

        #pdb.set_trace()

        dist = []

        for i in all_queues:
            dist.append([i[0], len(i[1])])
            if i[1] == []:
              continue
            else:
              for j in i[1]:
                avg_fid += (1-j[4])
                avg_wait += j[3]

        avg_fid_list.append(avg_fid/(CIRC_NUMBER*CIRC_CUT_FACTOR))
        avg_wait_list.append(avg_wait/CIRC_NUMBER)

        bar_plot(
            y=np.array([i[1] for i in dist]),
            bar_labels=[i[0] for i in dist],
            filename='dist{}.png'.format(str(fid)),
            y_integer=True,
            text='Avg fidelity: {}\nAvg waitting time:{}'.format(round(avg_fid/(CIRC_NUMBER*CIRC_CUT_FACTOR),4), round(avg_wait/(CIRC_NUMBER),3)),
            text_pos=(len(dist)+1, 10)
            )
    
    return all_all_queues, avg_fid_list, avg_wait_list


def final_plot(fid, wait):
    #line_data = pd.DataFrame()
    #print(FID_WEIGHTS)
    #print(avg_fid_list)
    #print(avg_wait_list)    
    #line_data['fid_weights'] = FID_WEIGHTS
    #line_data['avg_fid'] = avg_fid_list
    #line_data['avg_wait'] = avg_wait_list

    line_data = pd.DataFrame()
    print(FID_WEIGHTS, 'lll')
    line_data['fid_weights'] = FID_WEIGHTS
    sns.set_theme()
    colors = sns.color_palette("pastel")
    line_data['avg_wait'] = wait
    line_data['avg_fid'] = fid
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.legend(loc='upper left')
    sns.lineplot(data=line_data, x='fid_weights', y='avg_fid', ax=ax1, label='Avg fidelity', marker='o', color=colors[0])
    yticks = np.arange(min(fid), max(fid), (max(fid)-min(fid))/10)
    yticks = [round(i,3) for i in yticks]
    ax1.set_yticks(yticks)
    print(yticks)
    yticks = np.arange(min(wait), max(wait), (max(wait)-min(wait))/10)
    yticks = [round(i,3) for i in yticks]
    print(yticks)
    ax2.set_yticks(yticks)
    ax2.grid(None)
    sns.lineplot(data=line_data, x='fid_weights', y='avg_wait', ax=ax2, label='Avg waitting time(s)', marker='^', color=colors[1])
    ax1.set_ylabel('Avg fidelity(%)', color=colors[0])
    ax2.set_ylabel('Avg waitting time(s)', color=colors[1])
    ax2.legend(loc='upper right')
    ax1.set_xlabel('Fidelity weights')
    #plt.xticks(FID_WEIGHTS)
    save_figure(fig, 'test')

def line_plot(x, y, xlabel='XLabel', ylabel='YLabel',filename='filename', ticks_rounding=lambda x:x):

    line_data = pd.DataFrame()
    print(FID_WEIGHTS)
    line_data['x'] = x
    sns.set_theme()
    colors = sns.color_palette("pastel")
    line_data['y'] = y
    fig, ax1 = plt.subplots()
    sns.lineplot(data=line_data, x='x', y='y', ax=ax1, label=ylabel, marker='o', color=colors[0])
    yticks = np.arange(min(y), max(y), (max(y)-min(y))/10)
    yticks = [round(i,3) for i in yticks]
    yticks = [ticks_rounding(i) for i in yticks]
    ax1.set_yticks(yticks)
    ax1.legend(loc='upper left')
    ax1.set_ylabel(ylabel, color=colors[0])
    ax1.set_xlabel(xlabel)
    #plt.xticks(FID_WEIGHTS)
    save_figure(fig, filename)

#Info on Local queue 5 values
# qernel id
# estimated execution time
# submission time,
# estimated waiting time,
# predicted error

this = hard_job()
pdb.set_trace()
line_plot(FID_WEIGHTS, this[1], xlabel='Fidelity weights', ylabel='Avg fidelity', filename='avg_fid', ticks_rounding=lambda x:round(x,3))
line_plot(FID_WEIGHTS, this[2], xlabel='Fidelity weights', ylabel='Avg waiting time(s)', filename='avg_wait', ticks_rounding=lambda x:int(x))

#final_plot(this[1], this[2])