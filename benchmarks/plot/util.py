import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib import gridspec
import os
import pdb
import typing

FONTSIZE = 12
ISBETTER_FONTSIZE = FONTSIZE + 2
WIDE_FIGSIZE = (13, 2.8)
COLUMN_FIGSIZE = (6.5, 3.4)
COLUMN_FIGSIZE_2 = (7.5, 4)

plt.rcParams.update({"font.size": FONTSIZE})


def grouped_bar_plot(
    ax: plt.Axes,
    y: np.ndarray,
    yerr: np.ndarray,
    bar_labels: list[str],
    colors: list[str] | None = None,
    hatches: list[str] | None = None,
    show_average_text: bool = False,
    average_text_position: float = 1.05,
    spacing: float = 0.95,
    zorder: int = 2000,
):
    if colors is None:
        colors = sns.color_palette("pastel")
    if hatches is None:
        hatches = hatches = [
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

    #assert len(y.shape) == len(yerr.shape) == 2
    #assert len(y.shape) == 2
    #assert y.shape == yerr.shape

    num_groups, num_bars = y.shape
    #print(num_groups, len(bar_labels), num_bars)
 
    assert len(bar_labels) == num_bars
    

    bar_width = spacing / (num_bars + 1)
    x = np.arange(num_groups)

    for i in range(num_bars):
        y_bars = y[:, i]
        yerr_bars = yerr[:, i]

        color, hatch = colors[i % len(colors)], hatches[i % len(hatches)]

        #print(color)

        ax.bar(
            x + (i * bar_width),
            y_bars,
            bar_width,
            hatch=hatch,
            label=bar_labels[i],
            yerr=yerr_bars,
            color=color,
            edgecolor="black",
            linewidth=1.5,
            error_kw=dict(lw=2, capsize=3),
            zorder=zorder,
        )
    ax.set_xticks(x + ((num_bars - 1) / 2) * bar_width)

    if show_average_text:
        for i, x_pos in enumerate(ax.get_xticks()):
            y_avg = np.average(y[i])
            text = f"{y_avg:.2f}"
            print(text)
            ax.text(x_pos, average_text_position, text, ha="center")

def bar_plot(
    y: np.ndarray,
    bar_labels: list[str],
    colors: list[str] | None = None,
    hatches: list[str] | None = None,
    spacing: float = 2,
    zorder: int = 2000,
    filename: str = None,
    y_integer: bool = False,
    text=None,
    text_pos:tuple=None
    ):
    if colors is None:
        colors = sns.color_palette("pastel")
    if hatches is None:
        hatches = hatches = [
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

    #assert len(y.shape) == len(yerr.shape) == 2
    #assert len(y.shape) == 2
    #assert y.shape == yerr.shape

    num_bars = len(y)
    x = np.arange(num_bars)

    fig, ax = plt.subplots()

    color, hatch = colors[:len(y)], hatches[:len(y)]

    bar_width = spacing / (num_bars)

    plt.xticks(rotation=45)

    ax.bar(
        x,
        y,
        bar_width,
        hatch=hatch,
        tick_label=bar_labels,
        #yerr=yerr_bars,
        color=color,
        edgecolor="black",
        linewidth=1.5,
        error_kw=dict(lw=2, capsize=3),
        zorder=zorder,
    )
    if text != None:
        plt.text(*text_pos, text)

    if y_integer:
        y_ticks_integer = np.arange(0, max(y) + 1, (max(y) // 10) + 1)
        ax.set_yticks(ticks=y_ticks_integer)

    save_figure(fig, filename)

def index_dataframe_mean_std(
    df: pd.DataFrame,
    xkey: str,
    xvalues: np.ndarray,
    ykey: str,
) -> tuple[np.ndarray, np.ndarray]:
    df = (
        df.groupby(xkey)
        .agg({ykey: ["mean", "sem"]})
        .sort_values(by=[xkey])
        .reset_index()[[xkey, ykey]]
    )
    df = df.set_index(xkey)
    df = df.reindex(sorted(xvalues))
    df[ykey] = df[ykey].fillna(0.0)
    df = df.reset_index()
    return np.array(df[ykey]["mean"]), np.array(df[ykey]["sem"])


def data_frames_to_y_yerr(
    dataframes: list[pd.DataFrame],
    xkey: str,
    xvalues: np.ndarray,
    ykey: str,
    ykey_base: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if ykey_base is not None:
        for df in dataframes:
            df[ykey] = df[ykey] / df[ykey_base]

    mean_data, std_data = [], []
    for df in dataframes:
        mean, std = index_dataframe_mean_std(df, xkey, xvalues, ykey)
        mean_data.append(mean)
        std_data.append(std)

    return np.array(mean_data), np.array(std_data)


def save_figure(fig: plt.Figure, exp_name: str):
    plt.tight_layout()
    fig.savefig(
        exp_name + ".pdf",
        bbox_inches="tight",
    )


# if __name__ == "__main__":
#     from data import NOISE_SCALE_KOLKATA

#     dfs = [pd.read_csv(file) for file in NOISE_SCALE_KOLKATA.values()]
#     labels = list(NOISE_SCALE_KOLKATA.keys())

#     print(
#         index_dataframe_mean_std(
#             dfs[0], "num_qubits", np.array([8, 10, 12]), "num_cnots_base"
# #         )
#     ) 


def plot_lines_2yaxis(xkey:str, xlabel: str, ykeys: list[str], labels: list[str], data: pd.DataFrame, filename: str):

    sns.set_theme()
    sns.set_style("whitegrid")

    fig, ax1 = plt.subplots()
    sns.lineplot(data=data, x=xkey, y=ykeys[0], ax=ax1, label=labels[0], markers='o')

    ax2 = ax1.twinx()

    sns.lineplot(data=data, x=xkey, y=ykeys[1], ax=ax2, label=labels[1], markers='o')

    ax1.set_ylabel(labels[0], color='b')
    ax2.set_ylabel(labels[1], color='r')
    ax1.set_xlabel(xlabel)
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    save_figure(fig, filename)

def prepare_dataframe(df: pd.DataFrame, key: str) -> pd.DataFrame:
    res_df = df.loc[df[key] > 0.0]
    res_df = (
        res_df.groupby("num_qubits")
        .agg({key: ["mean", "sem"]})
        .sort_values(by=["num_qubits"])
        .reset_index())
    return res_df





# def calculate_figure_size(num_rows, num_cols):
#     subplot_width_inches = 3.0  # Adjust this value based on your desired subplot width

#     # Define the number of columns and rows of subplots
#     num_cols = 2
#     num_rows = 3

#     # Calculate the total width and height based on the subplot width and number of columns and rows
#     fig_width_inches = subplot_width_inches * num_cols
#     fig_height_inches = fig_width_inches / 1.618 * num_rows  # Incorporate the golden ratio (1.618) for the height

#     return fig_width_inches, fig_height_inches


if __name__ == "__main__":
    fig, ax = plt.subplots(1, figsize=COLUMN_FIGSIZE)

    x = np.array([15, 20, 25])
    y = np.array(
        [
            [0.5, 0.6],
            [0.4, 0.5],
            [0.3, 0.4],
        ]
    )
    yerr = np.array(
        [
            [0.5, 0.6],
            [0.4, 0.5],
            [0.3, 0.4],
        ]
    )
    ax.set_xticklabels(x)
    ax.grid(axis="y", linestyle="-", zorder=-1)
    grouped_bar_plot(ax, y, yerr, ["simtime", "knittime"])
    ax.legend(loc="upper right")
    save_figure(fig, "test")
