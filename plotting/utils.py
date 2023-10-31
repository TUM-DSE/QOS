import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

COLUMN_FIGSIZE = (6.5, 3.4)
COLUMN_FIGSIZE_2 = (7.5, 4)
WIDE_FIGSIZE = (13, 2.8)
FONTSIZE = 12
ISBETTER_FONTSIZE = FONTSIZE + 2
HIGHERISBETTER = "Higher is better ↑"
LOWERISBETTER = "Lower is better ↓"

line_markers = [
    "o",
    "v",
    "s",
    "^",
    "<",
    ">",
    "8",
    ]

def gen_subplots(ncols, nrows):
    fig = plt.figure(figsize=WIDE_FIGSIZE)
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    axes = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

    return fig, axes

def save_figure(fig: plt.Figure, exp_name: str):
    plt.tight_layout()
    fig.savefig(
        exp_name + ".pdf",
        bbox_inches="tight",
    )