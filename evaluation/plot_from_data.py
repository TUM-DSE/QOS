from plotting.plot import line_plot, line_plot_dashed
from evaluation.config import FID_WEIGHTS
from plotting.utils import save_figure, ISBETTER_FONTSIZE, HIGHERISBETTER, LOWERISBETTER, gen_subplots
import pdb
import numpy as np
# The save_data file has the following format:
#[all_all_queues, avg_fid_list, avg_wait_list]

filenames = ['results/scheduler_eval_data_8.txt',
             'results/scheduler_eval_data_16.txt',
             'results/scheduler_eval_data_24.txt',
             'results/scheduler_eval_data_32.txt']

legend = ['8 Clients',
          '16 Clients',
          '24 Clients',
          '32 Clients']

avg_fid_list = []
avg_wait_list = []

for i in filenames:
    f = open(i, 'r')
    raw = f.readlines()
    data = [eval(j) for j in raw][0]
    avg_fid_list.append(data[1][::-1])
    avg_wait_list.append(data[2][::-1])

#Because append puts it in the end
avg_fid_list = avg_fid_list[::-1]
avg_wait_list = avg_wait_list[::-1]

fig, axis = gen_subplots(1, 1)

x_axis = FID_WEIGHTS[::-1]

#line_plot(x_axis, avg_fid_list, xlabel='Fidelity Weight', ylabel='Avg. Fidelity', legend=legend, axis=axis[0], show_legend=False)
#line_plot_dashed(x_axis, avg_wait_list, xlabel='Fidelity Weight', ylabel='Avg. Waiting Time [s]', legend=legend, axis=axis[0], show_legend=False)

avg_fid_list = [avg_fid_list[3]]
avg_wait_list = [avg_wait_list[3]]
legend1 = ['Fidelity']
legend2 = ['Waiting Time']

line_plot(x_axis, avg_fid_list, xlabel='Fidelity Weight', ylabel='Avg. Fidelity', legend=legend1, axis=axis[0], show_legend=False)

axis2 = axis[0].twinx()

line_plot_dashed(x_axis, avg_wait_list, xlabel='Fidelity Weight', ylabel='Avg. Waiting Time [s]', legend=legend2, axis=axis2, show_legend=True)

#fig.text(0, 0.6, HIGHERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=ISBETTER_FONTSIZE, rotation=90)
#fig.text(1, 0.6, LOWERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=ISBETTER_FONTSIZE, rotation=90)

#fig.text(0.25, 1, 'a) Average Fidelity', ha="center", va="center", fontsize=12, fontweight="bold")
#fig.text(0.75, 1, 'b) Average Waiting Time', ha="center", va="center", fontsize=12, fontweight="bold")

axis[0].invert_xaxis()
handles,labels = axis[0].get_legend_handles_labels()

axis2.grid(False)
#axis2.set_yticks(np.linspace(300, 1100, 7))
#axis[0].set_yticks(np.linspace(round(axis[0].get_yticks()[0],2), round(axis[0].get_yticks()[-1],2), len(axis2.get_yticks())))

axis2.set_yticks(np.linspace(axis2.get_yticks()[0], axis2.get_yticks()[-1], len(axis[0].get_yticks())))

axis[0].legend(
    handles=handles,
    labels=labels,
    loc="lower center",
    bbox_to_anchor=(0.3, -0.25),
    ncol=9,
    frameon=False,
)

handles,labels = axis2.get_legend_handles_labels()

axis2.legend(
    handles=handles,
    labels=labels,
    loc="lower center",
    bbox_to_anchor=(0.7, -0.25),
    ncol=9,
    frameon=False,
)

save_figure(fig, 'results/avg_merged')