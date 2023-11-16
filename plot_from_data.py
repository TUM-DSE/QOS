from plotting.plot import line_plot
from evaluation.config import FID_WEIGHTS
from plotting.utils import save_figure, ISBETTER_FONTSIZE, HIGHERISBETTER, LOWERISBETTER, gen_subplots
import pdb
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
    avg_fid_list.append(data[1][::-1])
    avg_wait_list.append(data[2][::-1])

#Because append puts it in the end
avg_fid_list = avg_fid_list[::-1]
avg_wait_list = avg_wait_list[::-1]

fig, axis = gen_subplots(2, 1)

x_axis = FID_WEIGHTS[::-1]

line_plot(x_axis, avg_fid_list, xlabel='Fidelity weight', ylabel='Avg. fidelity', legend=legend, axis=axis[0], show_legend=False)
line_plot(x_axis, avg_wait_list, xlabel='Fidelity weight', ylabel='Avg. waiting time [s]', legend=legend, axis=axis[1], show_legend=False)

fig.text(0.25, 1.1, HIGHERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=ISBETTER_FONTSIZE)
fig.text(0.75, 1.1, LOWERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=ISBETTER_FONTSIZE)

fig.text(0.25, 1, 'a) Average Fidelity', ha="center", va="center")
fig.text(0.75, 1, 'b) Average waiting time', ha="center", va="center")


axis[0].invert_xaxis()
axis[1].invert_xaxis()

handles,labels = axis[0].get_legend_handles_labels()

fig.legend(
    handles=handles,
    labels=labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=9,
    frameon=False,
)

save_figure(fig, 'results/avg_merged')