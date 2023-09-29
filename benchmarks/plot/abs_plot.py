import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib import gridspec


def plot_abs(
    ax: plt.Axes, df: pd.DataFrame, xkey: str, xvalues: list[int], ykeys: list[str], labels: list[str], spa
):
    ax.grid(axis="y", linestyle="--")

    df = (
        df.groupby(xkey)
        .agg({ykey: ["mean", "sem"] for ykey in ykeys})
        .sort_values(by=[xkey])
        .reset_index()
    )

    df = df.set_index(xkey)

    df = df.reindex(sorted(xvalues))

    for ykey in ykeys:
        df[ykey] = df[ykey].fillna(0)

    df = df.reset_index()

    spacing = 0.95
    nums_qubits = sorted(list(xvalues))  # type: ignore
    bar_width = spacing / (len(ykeys) + 1)
    x = np.arange(len(nums_qubits))

    for i, ykey in enumerate(ykeys):
        y = df[ykey]["mean"]
        yerr = df[ykey]["sem"]
        if np.isnan(yerr).any():
            yerr = None

        color, hatch = colors[i % len(colors)], hatches[i % len(hatches)]

        ax.bar(
            x + (i * bar_width),
            y,
            bar_width,
            hatch=hatch,
            label=labels[i],
            yerr=yerr,
            color=color,
            edgecolor="black",
            linewidth=1.5,
            error_kw=dict(lw=2, capsize=3),
        )

    ax.set_xticks(x + ((ykeys) - 1) / 2) * bar_width)))
    ax.set_xticklabels(nums_qubits)
