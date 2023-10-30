from fileinput import filename
import pdb
from plotting.plot import line_plot, bar_plot, merge_plots
from evaluation.config import FID_WEIGHTS
# The save_data file has the following format:
#[all_all_queues, avg_fid_list, avg_wait_list]

filenames = ['results/scheduler_eval_data_8.txt',
             'results/scheduler_eval_data_16.txt',
             'results/scheduler_eval_data_24.txt',
             'results/scheduler_eval_data_32.txt']

legend = ['8 clients',
          '16 clients',
          '24 clients',
          '32 clients']

avg_fid_list = []
avg_wait_list = []

for i in filenames:
    f = open(i, 'r')
    raw = f.readlines()
    data = [eval(j) for j in raw][0]
    avg_fid_list.append(data[1])
    avg_wait_list.append(data[2])

#Because append puts it in the end
avg_fid_list = avg_fid_list[::-1]
avg_wait_list = avg_wait_list[::-1]


fig0 = line_plot(FID_WEIGHTS, avg_fid_list, xlabel='Fidelity weight', ylabel='Avg. fidelity', legend=legend ,filename='results/avg_fid_merged', higher_lower_isBetter='higher')
fig1 = line_plot(FID_WEIGHTS, avg_wait_list, xlabel='Fidelity weight', ylabel='Avg. waiting time [s]', legend=legend ,filename='results/avg_wait_merged', higher_lower_isBetter='lower')

merge_plots(fig0=fig0, fig1=fig1, filename='results/merged_plot.pdf')