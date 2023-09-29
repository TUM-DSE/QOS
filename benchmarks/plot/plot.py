from util import *

HIGHERISBETTER = "Higher is better ↑"
LOWERISBETTER = "Lower is better ↓"

from get_average import get_average



def plot_dep_min_stats() -> plt.Figure:
    DEP_MIN_DATA_3 = {
        "BV": "bench/results/greedy_dep_min/3/bv.csv",
        "VQE-1": "bench/results/greedy_dep_min/3/vqe_1.csv",
        "HS-2": "bench/results/greedy_dep_min/3/hamsim_2.csv",
        "TL-1": "bench/results/greedy_dep_min/3/twolocal_1.csv",
        "TL-2": "bench/results/greedy_dep_min/3/twolocal_2.csv",
        "TL-3": "bench/results/greedy_dep_min/3/twolocal_3.csv",
        "QAOA-B": "bench/results/greedy_dep_min/3/qaoa_b.csv",
        "QAOA-3": "bench/results/greedy_dep_min/3/qaoa_r3.csv",
        "QAOA-4": "bench/results/greedy_dep_min/3/qaoa_r4.csv",
    }

    dfs = [pd.read_csv(file) for file in DEP_MIN_DATA_3.values()]
    labels = list(DEP_MIN_DATA_3.keys())

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=WIDE_FIGSIZE, sharey=True)
    xvalues = [8, 16, 24]

    ax0.set_ylim(0, 1.2)

    y, yerr = data_frames_to_y_yerr(
        dfs, "num_qubits", np.array(xvalues), "num_deps", "num_deps_base"
    )
    grouped_bar_plot(ax0, y.T, yerr.T, labels, show_average_text=True)
    ax0.set_ylabel("Rel. Qubit Dependencies")
    ax0.set_title("(a) Qubit Dependencies", fontweight="bold", fontsize=FONTSIZE)
    ax0.set_xlabel("Number of Qubits")
    ax0.set_xticklabels(xvalues)
    _relative_plot(ax0)

    y, yerr = data_frames_to_y_yerr(
        dfs, "num_qubits", np.array(xvalues), "depth", "depth_base"
    )
    grouped_bar_plot(ax1, y.T, yerr.T, labels, show_average_text=True)
    ax1.set_ylabel("Rel. Circuit Depth")
    ax1.set_title("(b) Circuit Depth", fontweight="bold", fontsize=FONTSIZE)
    ax1.set_xlabel("Number of Qubits")
    ax1.set_xticklabels(xvalues)
    _relative_plot(ax1)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=10,
        frameon=False,
    )

    fig.text(
        0.51,
        0.9,
        LOWERISBETTER,
        ha="center",
        fontsize=ISBETTER_FONTSIZE,
        fontweight="bold",
        color="midnightblue",
    )

from util import calculate_figure_size, plot_lines, grouped_bar_plot, data_frames_to_y_yerr
from data import SWAP_REDUCE_DATA, DEP_MIN_DATA, NOISE_SCALE_ALGIERS_DATA, SCALE_SIM_TIME, SCALE_SIM_MEMORY


sns.set_theme(style="whitegrid", color_codes=True)
colors = sns.color_palette("deep")

plt.rcParams.update({"font.size": 12})


def plot_swap_reduce() -> None:
    dfs = [pd.read_csv(file) for file in SWAP_REDUCE_DATA.values()]
    titles = list(SWAP_REDUCE_DATA.keys())

    plot_dataframes(
        dataframes=dfs,
        keys=["num_cnots", "num_cnots_base"],
        labels=["Ours", "Baseline"],
        titles=titles,
        ylabel="Number of CNOTs",
        xlabel="Number of Qubits",
        output_file="figures/swap_reduce/cnot.pdf",
    )
    plot_dataframes(
        dataframes=dfs,
        keys=["depth", "depth_base"],
        labels=["Ours", "Baseline"],
        titles=titles,
        ylabel="Circuit Depth",
        xlabel="Number of Qubits",
        output_file="figures/swap_reduce/depth.pdf",
    )

    fig.text(
        0.51,
        0.98,
        HIGHERISBETTER,
        ha="center",
        fontsize=ISBETTER_FONTSIZE,
        fontweight="bold",
        color="midnightblue",
    )

def insert_column(df):
    df['total_runtime'] = df['run_time'] + df['knit_time']

    return df

def dataframe_out_of_columns(dfs, lines, columns):
    merged_df = pd.DataFrame()

    merged_df["num_qubits"] = dfs[0]["num_qubits"].copy()
    merged_df.set_index("num_qubits")

    for i,f in enumerate(dfs):
        merged_df[lines[i]] = f[columns].copy()

    #merged_df.reset_index(drop = True, inplace = True)
    merged_df.set_index("num_qubits", inplace = True)

    return merged_df

def plot_endtoend_runtimes():
	dfs = [pd.read_csv(file) for file in SCALE_SIM_TIME.values()]
	dfs_mem = [pd.read_csv(file) for file in SCALE_SIM_MEMORY.values()]
 
	lines = [s.split("-")[-1] for s in SCALE_SIM_TIME.keys()]

	titles = ["(a) Εnd-to-end Runtime", "(b) Runtime Breakdown", "(c) Memory Consumption"]

	dfs = [insert_column(i) for i in dfs]
	big_dfs = dataframe_out_of_columns(dfs, lines, ["total_runtime"])
	
	dfs_mem_new = pd.DataFrame()
	dfs_mem_new["num_qubits"] = dfs_mem[0]["num_qubits"].copy()
	dfs_mem_new["Baseline"] = dfs_mem[0]["h_fid"]
	dfs_mem_new["QVM"] = dfs_mem[0]["h_fid_base"]
	dfs_mem_new["CutQC"] = dfs_mem[0]["tv_fid"]
	dfs_mem_new.set_index("num_qubits", inplace = True)
	#print(dfs_mem_new)
	
	dfs_ratio = pd.DataFrame()
	dfs_ratio["qpu_size"] = [15, 20, 25]
	dfs_ratio.set_index("qpu_size")
	
	dfs_ratio["simulation"] = [d.loc[4].at['run_time'] for d in dfs]
	dfs_ratio["knitting"] = [d.loc[4].at['knit_time'] for d in dfs]
	
	keys = dfs_ratio.keys()
	keys = keys[1:]

	custom_plot_dataframes(
		dataframes=[big_dfs, dfs_ratio, dfs_mem_new],
		keys=[big_dfs.keys(), keys, dfs_mem_new.keys()],
		labels=[big_dfs.keys(), dfs_ratio["qpu_size"].tolist(), dfs_mem_new.keys()],
		titles=titles,
		ylabel=["Runtime [s]", "Runtime [s]", "Memory [GBs]"],
		xlabel=["Number of Qubits", "QPU Size (Number of Qubits)", "Number of Qubits"],
		output_file="figures/scale_sim/hamsim_1.pdf",
		logscale=True,
		nrows=1
	)	

hatches = [
    "/",
	"\\",
	"//",
	"\\\\",
	"x",
	".",
	",",
	"*",
	"o",
	"O",
	"+",
	"X",
	"s",
	"S",
	"d",
	"D",
	"^",
	"v",
	"<",
	">",
	"p",
	"P",
	"$",
	"#",
	"%",
]

def custom_plot_dataframes(
	dataframes: list[pd.DataFrame],
	keys: list[list[str]],
	labels: list[list[str]],
	titles: list[str],
	ylabel: list[str],
	xlabel: list[str],
	output_file: str = "noisy_scale.pdf",
	nrows: int = 2,
	logscale = False,
) -> None:
	ncols = len(dataframes)
	fig = plt.figure(figsize=[13, 3.2])
	gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

	axis = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]
	
	axis[0].set_yscale("log")
	axis[1].set_yscale("log")
	axis[2].set_yscale("log")

	#axis[2].set_xlim([10, 30])
	axis[1].set_ylim([1, 50000])
	#axis[2].set_ylim([10, 10 ** 20])
	#axis[2].set_yscale("log")

	for i, ax in enumerate(axis):
		ax.set_ylabel(ylabel=ylabel[i])
		ax.set_xlabel(xlabel=xlabel[i])
	
	#print(keys)
	plot_lines(axis[0], keys[0], labels[0], [dataframes[0]])
	axis[0].legend()		
	axis[0].set_title(titles[0], fontsize=12, fontweight="bold")
	
	plot_lines(axis[2], keys[2], labels[2], [dataframes[2]])
	axis[2].legend()		
	axis[2].set_title(titles[2], fontsize=12, fontweight="bold")

	num_vgates = dataframes[1]['qpu_size'].tolist()
	simulation = dataframes[1]['simulation'].tolist()
	knitting = dataframes[1]['knitting'].tolist()
	data = {
		"Simulation" : simulation,
		"Knitting" : knitting,
	}

	x = np.array([15, 20, 25])
	#x = np.arange(len(num_vgates))  # the label locations
	#width = 0.25  # the width of the bars
	#multiplier = 0
	y = np.array(
		[
			[9.52130384114571, 120.0079321230296, 801.0942367650568],
			[11.77336971112527, 726.3718322570203, 208.40429024997866],
			[1.7376548638567328, 5857.7779290829785, 305.2052580610034]
		]
	)

	yerr = np.array(
		[
			[1.3718322570203, 6.270605635945685, 41.68920839508064],
			[2.7376548638567328, 33.503638901049, 8.03563788096653],
			[0.2052580610034, 155.2813523421064, 22.93891781999264]
		]
	)
	
	axis[1].set_xticklabels(x)
	axis[1].grid(axis="y", linestyle="-", zorder=-1)	
	grouped_bar_plot(axis[1], y, yerr, ["Compilation", "Simulation", "Knitting"])
	axis[1].legend()

	axis[1].set_yticks(np.logspace(1, 5, base=10, num=5, dtype='int'))
	axis[1].set_title(titles[1], fontsize=12, fontweight="bold")
	
	fig.text(0.5, 1, "Lower is better ↓", ha="center", va="center", fontweight="bold", color="navy", fontsize=14)
	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	plt.tight_layout(pad=1)
	plt.savefig(output_file, bbox_inches="tight")

def plot_dep_min() -> None:
	dfs = [pd.read_csv(file) for file in DEP_MIN_DATA.values()]
	titles = list(DEP_MIN_DATA.keys())

	plot_dataframes(
		dataframes=dfs,
		keys=["num_cnots", "num_cnots_base"],
		labels=["Ours", "Baseline"],
		titles=titles,
		ylabel="Number of CNOTs",
		xlabel="Number of Qubits",
		output_file="figures/dep_min/cnot.pdf",
	)
	plot_dataframes(
		dataframes=dfs,
		keys=["depth", "depth_base"],
		labels=["Ours", "Baseline"],
		titles=titles,
		ylabel="Circuit Depth",
		xlabel="Number of Qubits",
		output_file="figures/dep_min/depth.pdf",
	)

	plot_dataframes(
		dataframes=dfs,
		keys=["h_fid", "h_fid_base"],
		labels=["Ours", "Baseline"],
		titles=titles,
		ylabel="Fidelity",
		xlabel="Number of Qubits",
		output_file="figures/dep_min/fid.pdf",
	)


def plot_noisy_scale() -> None:
	dfs = [pd.read_csv(file) for file in NOISE_SCALE_ALGIERS_DATA.values()]
	titles = list(NOISE_SCALE_ALGIERS_DATA.keys())

	plot_dataframes(
		dataframes=dfs,
		keys=["num_cnots", "num_cnots_base"],
		labels=["Dep Min", "Baseline"],
		titles=titles,
		ylabel="Number of CNOTs",
		xlabel="Number of Qubits",
		output_file="figures/noisy_scale/algiers_cnot.pdf",
		nrows=3,
	)
	plot_dataframes(
		dataframes=dfs,
		keys=["depth", "depth_base"],
		labels=["Dep Min", "Baseline"],
		titles=titles,
		ylabel="Circuit Depth",
		xlabel="Number of Qubits",
		output_file="figures/noisy_scale/algiers_depth.pdf",
		nrows=3,
	)
	plot_dataframes(
		dataframes=dfs,
		keys=["h_fid", "h_fid_base"],
		labels=["Ours", "Baseline"],
		titles=titles,
		ylabel="Fidelity",
		xlabel="Number of Qubits",
		output_file="figures/noisy_scale/algiers_fid.pdf",
		nrows=3,
	)

def plot_dataframes(
	dataframes: list[pd.DataFrame],
	keys: list[str],
	labels: list[str],
	titles: list[str],
	ylabel: str,
	xlabel: str,
	output_file: str = "noisy_scale.pdf",
	nrows: int = 2,
	logscale = False,
) -> None:
	ncols = len(dataframes) // nrows + len(dataframes) % nrows
	# plotting the absolute fidelities
	fig = plt.figure(figsize=calculate_figure_size(nrows, ncols))
	gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

	axis = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

	for i, ax in enumerate(axis):
		if logscale:
			ax.set_yscale("log")
		if i % ncols == 0:
			ax.set_ylabel(ylabel=ylabel)
		if i >= len(axis) - ncols:
			ax.set_xlabel(xlabel=xlabel)

	for let, title, ax, df in zip(string.ascii_lowercase, titles, axis, dataframes):
		plot_lines(ax, keys, labels, [df])
		ax.legend()
		ax.set_title(f"({let}) {title}", fontsize=12, fontweight="bold")

	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	plt.tight_layout()
	plt.savefig(output_file, bbox_inches="tight")


def plot_relative(
	ax,
	dataframes: list[pd.DataFrame],
	num_key: str,
	denom_key: str,
	labels: list[str],
	ylabel: str,
	xlabel: str,
	title: str | None = None,
):
	for df in dataframes:
		df["relative"] = df[num_key] / df[denom_key]

	plot_lines(ax, ["relative"], labels, dataframes)
	ax.set_ylabel(ylabel=ylabel)
	ax.set_xlabel(xlabel=xlabel)

	# line at 1
	ax.axhline(y=1, color="black", linestyle="--")
	ax.set_title(title)
	ax.legend()


def plot_relative_swap_reduce() -> None:
	DATAFRAMES = [pd.read_csv(file) for file in SWAP_REDUCE_DATA.values()]
	LABELS = list(SWAP_REDUCE_DATA.keys())

	fig, ax = plt.subplots(1, 1, figsize=calculate_figure_size(1, 1))
	plot_relative(
		ax,
		DATAFRAMES,
		"num_cnots",
		"num_cnots_base",
		labels=LABELS,
		ylabel="Reulative Number of CNOTs",
		xlabel="Number of Qubits",
	)

	output_file = "figures/swap_reduce/cnot_relative.pdf"

	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	plt.tight_layout()
	plt.savefig(output_file, bbox_inches="tight")



def main():
	#print("Plotting swap reduce...")
	#plot_swap_reduce()
	#print("Plotting dep min...")
	#plot_dep_min()
	#print("Plotting noisy scale...")
	#plot_noisy_scale()
	# plot_relative_swap_reduce()
	plot_endtoend_runtimes()


if __name__ == "__main__":
	main()
