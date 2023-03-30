import pdb
import matplotlib.pyplot as plt
import numpy as np

# Define a matrix of values
"""
matrix = np.array(
    [
        ["Hamiltonian", 82],
        ["VanillaQAOA", 29],
        ["GHZ", 28],
        ["VQE", 78],
        ["BitCode", 27],
        ["PhaseCode", 29],
        ["MerminBell", 21],
        ["FermionicSwap", 43],
    ]
)
"""

matrix = np.array(
    [
        ["Hamiltonian", 90],
        ["VanillaQAOA", 87],
        ["GHZ", 85],
        ["VQE", 87],
        ["BitCode", 80],
        ["PhaseCode", 80],
        ["MerminBell", 69],
        ["FermionicSwap", 59],
    ]
)
print(matrix)

# pdb.set_trace()
graph = plt.bar(matrix[:, 0], matrix[:, 1].astype(float))

# Set the y-label
plt.ylim([0, 100])
plt.xticks(rotation=25)

for i, bar in enumerate(graph):
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height(),
        str(matrix[i, 1]),
        ha="center",
        va="bottom",
    )

plt.axhline(
    y=np.median(matrix[:, 1].astype(float)), color="g", linestyle="--", label="Avg"
)

plt.ylabel("Utilization (%)")
plt.xlabel("Benchmark")
plt.title("Utilization of the QPU for a threshold of 85% fidelity")
# Show the plot
plt.rcParams["savefig.dpi"] = 256
plt.tight_layout()
# Save the heatmap to a file
plt.savefig("bar_plot.png")
plt.show()
