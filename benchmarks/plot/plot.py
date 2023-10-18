from benchmarks.plot.util import *

HIGHERISBETTER = "Higher is better ↑"
LOWERISBETTER = "Lower is better ↓"

#from get_average import get_average



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

#from util import calculate_figure_size, plot_lines, grouped_bar_plot, data_frames_to_y_yerr
#from data import SWAP_REDUCE_DATA, DEP_MIN_DATA, NOISE_SCALE_ALGIERS_DATA, SCALE_SIM_TIME, SCALE_SIM_MEMORY


sns.set_theme(style="whitegrid", color_codes=True)
colors = sns.color_palette("deep")
plt.rcParams.update({"font.size": 12})

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

def custom_plot_multiprogramming(	
	titles: list[str],
	ylabel: list[str],
	xlabel: list[str],
	output_file: str = "multi_programming.pdf",
) -> None:
	fig = plt.figure(figsize=COLUMN_FIGSIZE)

	x = np.array([59, 74, 88])

	nrows = 1
	ncols = 1
	gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

	axis = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

	axis[0].set_ylabel(ylabel[0])
	axis[0].set_xlabel(xlabel[0])
	axis[0].set_ylim(0, 1)

	y = np.array(
		[
			[0.81, 0.88, 0.94],
			[0.68, 0.83, 0.945],
			[0.53, 0.78, 0.92]
		]
	)

	yerr = np.array(
		[
			[0, 0, 0],
			[0, 0, 0],
			[0, 0, 0]
		]
	)
	
	axis[0].set_xticklabels(x)
	axis[0].grid(axis="y", linestyle="-", zorder=-1)	
	grouped_bar_plot(axis[0], y, yerr, ["No M/P", "Random M/P", "QOS M/P"])
	axis[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3)

	#axis[0].set_yticks(np.logspace(1, 5, base=10, num=5, dtype='int'))
	axis[0].set_title(titles[0], fontsize=FONTSIZE, fontweight="bold")
	
	fig.text(0.5, 1, HIGHERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=ISBETTER_FONTSIZE)

	#os.makedirs(os.path.dirname(output_file), exist_ok=True)
	#plt.tight_layout(pad=1)
	plt.savefig(output_file, bbox_inches="tight")

def custom_plot_multiprogramming_relative(titles: list[str],
	ylabel: list[str],
	xlabel: list[str],
	output_file: str = "multi_programming_relative.pdf",):

	fig = plt.figure(figsize=WIDE_FIGSIZE)

	x = np.array([59, 74, 88])

	nrows = 1
	ncols = 1
	gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

	axis = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

	axis[0].set_ylabel(ylabel[0])
	axis[0].set_xlabel(xlabel[0])
	axis[0].set_ylim(0, 1.1)

	y = np.array(
		[
			[0.98, 0.933, 0.94, 0.97, 0.98, 0.9456, 0.943, 0.96, 0.9452],
			[0.945, 0.914, 0.92, 0.951, 0.967, 0.929, 0.918, 0.948, 0.923],
			[0.913, 0.905, 0.903, 0.934, 0.938, 0.905, 0.897, 0.93, 0.901]
		]
	)

	yerr = np.array(
		[
			[0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0]
		]
	)

	axis[0].set_xticklabels(x)
	axis[0].grid(axis="y", linestyle="-", zorder=-1)

	grouped_bar_plot(axis[0], y, yerr, ["W-State", "QSV", "TL-1", "HS-1", "HS-2", "VQE-1", "VQE-2", "QAOA-B", "QAOA-2"], show_average_text=True, average_text_position=1.03)

	axis[0].axhline(1, color="red", linestyle="-", linewidth=2)

	#axis[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3)
	handles, labels = axis[0].get_legend_handles_labels()

	fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=9,
        frameon=False,
    )

	#axis[0].set_yticks(np.logspace(1, 5, base=10, num=5, dtype='int'))
	axis[0].set_title(titles[0], fontsize=FONTSIZE, fontweight="bold")
	
	fig.text(0.5, 1, HIGHERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=ISBETTER_FONTSIZE)

	#os.makedirs(os.path.dirname(output_file), exist_ok=True)
	#plt.tight_layout(pad=1)
	plt.savefig(output_file, bbox_inches="tight")


def custom_plot_large_circuit_fidelities(dataframes: list[pd.DataFrame], titles: list[str],
	ylabel: list[str],
	xlabel: list[str],
	output_file: str = "scalability_results.pdf"):

	fig = plt.figure(figsize=WIDE_FIGSIZE)

	x = np.array([4, 8, 12, 16, 20, 24])
	#x = dataframes[0]["bench_name"]

	nrows = 1
	ncols = 1
	gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

	axis = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

	axis[0].set_ylabel(ylabel[0])
	axis[0].set_xlabel(xlabel[0])
	axis[0].set_ylim(0, 1.1)

	y = np.array(
		[
			df["fidelity"] for df in dataframes
		]
	)

	yerr = np.array(
		[
			df["fidelity_std"] for df in dataframes
		]
	)
	

	axis[0].set_xticklabels(x)
	axis[0].grid(axis="y", linestyle="-", zorder=-1)

	grouped_bar_plot(axis[0], y, yerr, ["QAOA-R3", "BV", "GHZ", "HS-1", "QAOA-P1", "QSVM", "TL-1", "VQE-1", "W-STATE"], show_average_text=True, average_text_position=1.03)

	handles, labels = axis[0].get_legend_handles_labels()

	fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=9,
        frameon=False,
    )

	axis[0].set_title(titles[0], fontsize=FONTSIZE, fontweight="bold")
	
	fig.text(0.5, 1, HIGHERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=ISBETTER_FONTSIZE)

	#os.makedirs(os.path.dirname(output_file), exist_ok=True)
	#plt.tight_layout(pad=1)
	plt.savefig(output_file, bbox_inches="tight")


def custom_plot_small_circuit_relative_fidelities(dataframes: list[pd.DataFrame], titles: list[str],
	ylabel: list[str],
	xlabel: list[str],
	output_file: str = "scalability_proposal.pdf"):

	fig = plt.figure(figsize=COLUMN_FIGSIZE_2)

	x = np.array(["QAOA-R3", "BV", "GHZ", "HS-1", "QAOA-P1", "QSVM", "TL-1", "VQE-1", "W-STATE"])
	#x = dataframes[0]["bench_name"]

	nrows = 1
	ncols = 1
	gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

	axis = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

	axis[0].set_ylabel(ylabel[0])
	axis[0].set_xlabel(xlabel[0])
	axis[0].set_yscale("log", base=2)
	axis[0].set_ylim(1, 32)

	#ytick_locations = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
	ytick_labels = ['0', '1', '2', '4', '8', '16', '32']
	axis[0].set_yticklabels(ytick_labels)

	#axis[0].axhline(1, color="red", linestyle="-", linewidth=2)
	

	y = np.array(
		[
			dataframes[0]["fidelity"] / dataframes[1]["fidelity"]
		]
	)

	yerr = np.array(
		[
			dataframes[0]["fidelity_std"] / dataframes[1]["fidelity_std"]
		]
	)
	
	#print(y)
	#print(yerr)
	axis[0].set_xticklabels(x)
	axis[0].grid(axis="y", linestyle="-", zorder=-1)

	grouped_bar_plot(axis[0], y.T, yerr.T, [""], show_average_text=True, average_text_position=24)

	axis[0].set_title(titles[0], fontsize=FONTSIZE, fontweight="bold")
	
	fig.text(0.5, 0.95, HIGHERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=ISBETTER_FONTSIZE)

	plt.savefig(output_file, bbox_inches="tight")


def custom_plot_small_circuit_relative_properties(dataframes: list[pd.DataFrame], titles: list[str],
	ylabel: list[str],
	xlabel: list[str],
	output_file: str = "relative_properties.pdf"):

	fig = plt.figure(figsize=WIDE_FIGSIZE)

	x = np.array([12, 24])
	#x = dataframes[0]["bench_name"]

	nrows = 1
	ncols = 2
	gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

	axis = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

	axis[0].set_ylabel(ylabel[0])
	axis[0].set_xlabel(xlabel[0])

	axis[1].set_ylabel(ylabel[1])
	axis[1].set_xlabel(xlabel[1])

	axis[0].set_ylim(0, 2)
	axis[1].set_ylim(0, 2)

	axis[0].axhline(1, color="red", linestyle="-", linewidth=2)
	axis[1].axhline(1, color="red", linestyle="-", linewidth=2)
	

	y0 = np.array(
		[
			df["depth"] for df in dataframes
		]
	)

	
	yerr0 = np.array(
		[
			df["fidelity"] for df in dataframes
		]
	)

	y1 = np.array(
		[
			df["num_nonlocal_gates"] for df in dataframes
		]
	)

	yerr1 = np.array(
		[
			df["fidelity"] for df in dataframes
		]
	)
	
	axis[0].set_xticklabels(x)
	axis[1].set_xticklabels(x)
	axis[0].grid(axis="y", linestyle="-", zorder=-1)
	axis[1].grid(axis="y", linestyle="-", zorder=-1)

	grouped_bar_plot(axis[0], y0, yerr0, ["QAOA-R3", "BV", "GHZ", "HS-1", "QAOA-P1", "QSVM", "TL-1", "VQE-1", "W-STATE"], show_average_text=True, average_text_position=1.6)
	grouped_bar_plot(axis[1], y1, yerr1, ["QAOA-R3", "BV", "GHZ", "HS-1", "QAOA-P1", "QSVM", "TL-1", "VQE-1", "W-STATE"], show_average_text=True, average_text_position=1.6)

	axis[0].set_title(titles[0], fontsize=FONTSIZE, fontweight="bold")
	axis[1].set_title(titles[1], fontsize=FONTSIZE, fontweight="bold")

	handles, labels = axis[0].get_legend_handles_labels()

	fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=9,
        frameon=False,
    )
	
	fig.text(0.5, 0.95, LOWERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=ISBETTER_FONTSIZE)

	plt.savefig(output_file, bbox_inches="tight")

def custom_plot_small_circuit_overheads(dataframes: list[pd.DataFrame], titles: list[str],
	ylabel: list[str],
	xlabel: list[str],
	output_file: str = "overheads.pdf"):

	fig = plt.figure(figsize=COLUMN_FIGSIZE)

	x = np.array([12, 24])
	#x = dataframes[0]["bench_name"]

	nrows = 1
	ncols = 1
	gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

	axis = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

	axis[0].set_ylabel(ylabel[0])
	axis[0].set_xlabel(xlabel[0])

	#axis[1].set_ylabel(ylabel[1])
	#axis[1].set_xlabel(xlabel[1])

	#axis[0].set_yscale("log")
	axis[0].set_ylim(0, 4)

	#ytick_labels = ['0', '1', '2', '4', '8', '16', '32']
	#axis[0].set_yticklabels(ytick_labels)

	#axis[0].axhline(1, color="red", linestyle="-", linewidth=2)
	#axis[1].axhline(1, color="red", linestyle="-", linewidth=2)
	
	to_plot = []
	#print(np.mean([df["num_nonlocal_gates"].to_list() for df in dataframes]))
	to_plot.append([])
	to_plot[0].append(np.mean(dataframes[0]["num_nonlocal_gates"].to_list()))
	to_plot[0].append(np.mean(dataframes[0]["num_measurements"].to_list()))
	to_plot[0].append(np.mean(dataframes[0]["fidelity"].to_list()))

	to_plot.append([])
	to_plot[1].append(np.mean(dataframes[1]["num_nonlocal_gates"].to_list()))
	to_plot[1].append(np.mean(dataframes[1]["num_measurements"].to_list()))
	to_plot[1].append(np.mean(dataframes[1]["fidelity"].to_list()))

	#print(to_plot)
	y0 = np.array(
		to_plot
	)

	
	yerr0 = np.array(
		[
			[0, 0, 0],
			[0, 0, 0]
		]
	)

	#print(y0)
	#exit()
	
	axis[0].set_xticklabels(x)
	axis[0].grid(axis="y", linestyle="-", zorder=-1)


	grouped_bar_plot(axis[0], y0, yerr0, ["Baseline", "QOS DT", "QOS Optimizer"])

	axis[0].set_title(titles[0], fontsize=FONTSIZE, fontweight="bold")
	#axis[1].set_title(titles[1], fontsize=FONTSIZE, fontweight="bold")
	#fig.legend()

	handles, labels = axis[0].get_legend_handles_labels()

	fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )
	
	fig.text(0.5, 0.95, LOWERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=ISBETTER_FONTSIZE)

	plt.savefig(output_file, bbox_inches="tight")

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
