from ast import List, Tuple
from cProfile import label
import pdb
from .utils import *
import pandas as pd
import seaborn as sns
import numpy as np
from .utils import *
import matplotlib.pyplot as plt
from .utils import COLUMN_FIGSIZE
from matplotlib import gridspec


def line_plot(x, y, xlabel='XLabel', ylabel='YLabel', legend:str|list[str]=None, show_legend=True, axis=None):
    
    if axis == None:
        fig = plt.figure(figsize=COLUMN_FIGSIZE)
        nrows = 1
        ncols = 1
        gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
        ax = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)][0]
    else:
        ax = axis

    sns.set_theme()
    sns.set_style("whitegrid")
    ax.grid(True)

    if isinstance(y, list):
        line_data = pd.DataFrame()
        line_data['x'] = x
        line_data.set_index('x', inplace=True)
        colors = sns.color_palette("pastel")
        ax.set_xlabel(xlabel, color='black')
        ax.set_ylabel(ylabel, color='black')

        for i in range(len(y)):
            if legend == None:
                sns.lineplot(x=x, y=y[i], ax=ax, marker=line_markers[i], color=colors[i], dashes=False)
            else:
                sns.lineplot(x=x, y=y[i], ax=ax, marker=line_markers[i], color=colors[i], label=legend[i], dashes=False, legend=False if not show_legend else True)

    else:
        line_data = pd.DataFrame()
        line_data['x'] = x
        sns.set_theme()
        sns.set_style("whitegrid")
        colors = sns.color_palette("pastel")
        line_data['y'] = y
        fig, ax1 = plt.subplots()
        sns.lineplot(data=line_data, x='x', y='y', ax=ax1, label=ylabel, marker='o', color=colors[1])
        yticks = np.arange(min(y), max(y), (max(y)-min(y))/10)
        ax1.set_yticks(yticks)
        ax1.legend(loc='upper left')
        ax1.set_ylabel(ylabel, color='black')
        ax1.set_xlabel(xlabel, color='black')

    if axis == None:
        return fig


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
    plt.close()



def merge_plots(fig0:plt.Figure, fig1:plt.Figure, filename:str):
    fig, ax = plt.subplots(ncols=2, figsize=WIDE_FIGSIZE, sharey=True)
    fig.add_subplot(fig0)
    fig.add_subplot(fig1)
    save_figure(fig, filename)
    plt.close()
