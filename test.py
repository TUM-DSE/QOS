import json
from datetime import datetime
from qiskit import IBMQ
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.colors import CSS4_COLORS as fancy_colors
import matplotlib.patches as mpatches

def datetime_to_str(obj):
    """Helper function to convert datetime objects to strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"{type(obj)} not serializable")


def convert_dict_to_json(d, file_path):
    """Recursively convert a dictionary with datetime objects to a JSON file"""
    # Recursively convert nested dictionaries
    for key, value in d.items():
        if isinstance(value, dict):
            convert_dict_to_json(value, file_path)

    # Convert datetime objects to strings using helper function
    d_str = json.dumps(d, default=datetime_to_str, indent=4)

    # Write JSON string to file
    with open(file_path, 'w') as f:
        f.write(d_str)
        

def get_callibration_data():  
    provider = IBMQ.load_account()
    backends = provider.backends()
    backend = provider.get_backend("ibm_lagos")

    for i in range(1, 12):
        for j in range(1, 28):
            t = datetime(day=j, month=i, year=2022, hour=10)

            properties = backend.properties(datetime=t)
            
            if properties is None:
                continue
                
            properties = properties.to_dict()

            convert_dict_to_json(properties, "callibration_data/ibm_lagos" + datetime_to_str(t) + ".json")

def read_data(filename):
    with open(filename, 'r') as f:
        numbers = [float(line.strip()) for line in f]

    return numbers

def plot_line(data, filename):
    x = np.arange(len(data))
    y = np.array(data)

    # Calculate the maximum difference between two consecutive values
    diff_consecutive = np.max(np.abs(np.diff(y)))

    # Calculate the maximum difference between any two values
    diff_all = np.max(np.abs(np.subtract.outer(y, y)))
    
    
    plt.axis([1, 300, 0.5, 1])
    # Plot the data
    plt.plot(x, y)

    # Add labels and title to the plot
    plt.xlabel('Callibration Day')
    plt.ylabel('Fidelity Score')
    #plt.title('GHZ score on IBMQ Lagos across 300 days')
    plt.axhline(np.median(y), color='green', linewidth=2)
    # Add text to the plot showing the maximum differences
    plt.text(0.01, 0.95, "Max single-day difference: {:.2f}".format(diff_consecutive),
             transform=plt.gca().transAxes)
    plt.text(0.01, 0.85, "Max difference: {:.2f}".format(diff_all),
             transform=plt.gca().transAxes)

    # Show the plot
    plt.savefig(filename, dpi=300)
    

def plot_bar_chart(filename, title, xlabel, ylabel):
    with open(filename, 'r') as file:
        data = [line.strip().split() for line in file]

    # separate x and y data
    #x_data = [item[0] for item in data]
    x_data = ["Random", "Best\navailable", "Max. fidelity\ndifference", "Best\nmachine", "Optimal"]
    y_data = [float(item[0]) for item in data]

    fig, ax = plt.subplots()
    bar_list = ax.bar(x_data, y_data)
    bar_list[0].set_color(fancy_colors["darkseagreen"])
    bar_list[0].set_edgecolor("k")
    #bar_list[0].set_hatch("-")
    bar_list[1].set_color(fancy_colors["darkseagreen"])
    bar_list[1].set_edgecolor("k")
    #bar_list[1].set_hatch("-")
    bar_list[2].set_color(fancy_colors["darkseagreen"])
    bar_list[2].set_edgecolor("k")
    #bar_list[2].set_hatch("-")
    bar_list[3].set_color(fancy_colors["lightcoral"])
    bar_list[3].set_edgecolor("k")
    #bar_list[3].set_hatch("x")
    bar_list[4].set_color(fancy_colors["lightcoral"])
    bar_list[4].set_edgecolor("k")
    #bar_list[4].set_hatch("x")
    #plt.xticks(rotation='vertical')
    plt.text(0.01, 0.95, "Manual vs automated: 0.02",
             transform=plt.gca().transAxes)
    
    plt.ylim(0.75, 1)
    # set axis labels and title
    red_patch = mpatches.Patch(color='darkseagreen', label='Automated')
    blue_patch = mpatches.Patch(color='lightcoral', label='Manual')

    plt.legend(handles=[red_patch, blue_patch])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_title(title)

    # show plot
    plt.savefig('selection.png', dpi=300, bbox_inches="tight")
    #plt.savefig('selection.png', dpi=300)
    
    
def plot_multi_data(filename):
    labels = []
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        
        for row in reader:
            labels.append(row[0])
            data.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])

    #x_axis_data = [d[0] for d in data]
    
    #means = [np.mean(data[name]) for name in data]
    #stds = [np.std(data[name]) for name in data]
    
    #labels = [str(i) for i in data.keys()]
    #coords = [i for i in range(5)]
    #print(header[1:])
    markers = ['o', '^', 's', 'p', 'd']
    #fig, ax = plt.subplots()
    for i,d in enumerate(data):
        plt.plot(header[1:], d, marker=markers[i], label=labels[i])
    
    plt.legend(title="Width diff.")
    plt.xlabel(header[0])
    plt.ylabel('Idle time (%)')
    #ax.set_title('GHZ fidelity score across IBMQ Backends')
    plt.savefig('multiprogramming.png', dpi=300, bbox_inches="tight")
    
def plot_data(filename):
    data = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        data = {int(row[0]) : [float(x) for x in row[1].split(',')] for row in reader}

    #x_axis_data = [d[0] for d in data]
    
    means = [np.mean(data[name]) for name in data]
    stds = [np.std(data[name]) for name in data]
    
    labels = [str(i) for i in data.keys()]
    coords = [i for i in range(5)]

    fig, ax = plt.subplots()
    ax.bar(coords, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10, tick_label=labels)
    #ax.set_xticks(coords)
    ax.set_xlabel('Maximum fragment size (# of qbits)')
    ax.set_ylabel('Fidelity Score')
    #ax.set_title('GHZ fidelity score across IBMQ Backends')
    plt.savefig('plot.png', dpi=300, bbox_inches="tight")
#plot_bar_chart('results.txt', 'GHZ score across IBMQ Backends', 'IBMQ Backend', 'Benchmark Score')


def plot_benchmarks():
    bar_labels = ["3q", "55", "7q", "9q", "11q"]
    group_labels = ["Hamiltonian", "VQE", "VanillaQAOA", "GHZ", "BitCode", "PhaseCode", "FermionicQAOA", "MerminBell"]
    scores = {
    "3" : [0.9688343047,0.9922644065,0.7519531344,0.7744149943,0.9364013672,0.6741943359,0.5746459961,0.6619485715], 
    "5" : [0.9411731719,0.9320017355,0.5647793139,0.6501866229,0.9462890625,0.6351318359,0.5147094727,0.5206707622], 
    "7": [0.917268317,0.8661151583,0.4901475463,0.5041666578,0.8515625,0.6157470703,0.5043334961,0.5045776367],
    "9": [0.9311959584,0.7815148471,0.4612598637,0.3938107102,0.6522216797,0.5932617188,0.5015869141,0.4977636719],
    "11": [0.945296929,0.7852776514,0.439502009,0.3659576413,0.4794921875,0.4569091797,0.4165347388,0.469648994]}
    
    x = np.arange(len(group_labels))  # the label locations
    width = 0.13  # the width of the bars
    multiplier = 0

    #fig, ax = plt.subplots(layout='constrained')
    fig, ax = plt.subplots()

    for qbits, score in scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=qbits)
        #ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_xlabel("Benchmarks")
    ax.set_ylabel("Fidelity Score")
    ax.set_xticks(x + 2 * width, group_labels, rotation=90)
    # Add title and legend
    #ax.set_title("Benchmark score with increasing number of qubits")
    ax.legend()
        
    plt.savefig('plot.png', dpi=300, bbox_inches="tight")

#data = read_data('results.txt')
#plot_line(data, 'plot.png')

plot_bar_chart("selection.txt", "", "Backend selection scheme", "Fidelity score")