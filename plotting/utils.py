import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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


def save_figure(fig: plt.Figure, exp_name: str):
    plt.tight_layout()
    fig.savefig(
        exp_name + ".pdf",
        bbox_inches="tight",
    )