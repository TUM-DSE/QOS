from typing import List
from qos.kernel.matcher import Matcher
from qos.kernel.scheduler import Scheduler
import os
import logging
import pdb
import qos.database as db
import numpy as np
from evaluation.util import *
from evaluation.config import CIRC_NUMBER, CIRC_CUT_FACTOR, FID_WEIGHTS
from plotting.plot import line_plot, bar_plot

def main():
    try:
        os.system(f'redis-cli flushall')   
        os.system(f'python load_qpus.py')
    except FileNotFoundError:
        print(f"Error: The file does not exist.")

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
            filename='results/dist{}.png'.format(str(round(fid,3))),
            y_integer=True,
            text='Avg fidelity: {}\nAvg waitting time:{}'.format(round(avg_fid/(CIRC_NUMBER*CIRC_CUT_FACTOR),4), round(avg_wait/(CIRC_NUMBER),3)),
            text_pos=(len(dist)+1, 10)
            )

    save_data = open('results/scheduler_eval_data.txt', 'w')
    save_data.write(str([all_all_queues, avg_fid_list, avg_wait_list]))

    line_plot(FID_WEIGHTS, avg_fid_list, xlabel='Fidelity weights', ylabel='Avg fidelity', filename='results/avg_fid', ticks_rounding=lambda x:x)
    line_plot(FID_WEIGHTS, avg_wait_list, xlabel='Fidelity weights', ylabel='Avg waiting time(s)', filename='results/avg_wait', ticks_rounding=lambda x:int(x))

    return

main()