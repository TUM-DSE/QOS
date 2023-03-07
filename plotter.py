import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pdb
import sys

# create an input argv option for the word static
if sys.argv[1] == "static":

    csv_file = open('results/results.csv', 'r')
    
    csv_reader = csv.reader(csv_file)
    headers = next(csv_reader)  # Skip the header row
    data = [[row[0], int(row[1]), float(row[5])-float(row[4])] for row in csv_reader]

    # Create the bar chart
    fig, ax = plt.subplots()
    index = range(len(data))
    bar_width = 0.8
    opacity = 0.8
    rects1 = ax.bar(index, [x[2] for x in data], bar_width,
                    alpha=opacity, color='b')

    # Add labels, titles, and legend
    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Difference')
    ax.set_title('Difference between CNOT counts')
    ax.set_xticks(index)

    # Switch case for the following benchmark names
    # Hamiltonian = Ham
    # VQE = VQE
    # BitCode = Bit
    # PhaseCode = Phase
    # QAOA = QAOA
    # Mermin = Bell
    # GHZ = GHZ
    # Fermionic = Fermi
    labels = []

    for row in data:
        if row[0] == 'Hamiltonian':
            labels.append("Ham")
        elif row[0] == 'VQE':
            labels.append("VQE")
        elif row[0] == 'BitCode':
            labels.append("Bit")
        elif row[0] == 'PhaseCode':
            labels.append("Phase")
        elif row[0] == 'QAOA':
            labels.append("QAOA")
        elif row[0] == 'MerminBell':
            labels.append("Bell")
        elif row[0] == 'GHZ':
            labels.append("GHZ")
        elif row[0] == 'FermionicQAOA':
            labels.append("Fermi")

    ax.set_xticklabels(labels)
    

    # Display the chart
    plt.savefig('results/bar_chart.png', dpi=300, bbox_inches='tight')
    csv_file.close()

else:

    file = open('results/results.csv', 'r')
    reader = csv.reader(file)
    data = list(reader)

# Get the header row and remove it from the data
    header = data[0]
    data = data[1:]
    # Get the unique values of bench1 and bench2
    bench1_values = list(set([row[0] for row in data]))
    bench2_values = list(set([row[3] for row in data]))

    # Create a 2D array to store the bench1_fid values
    fid_values = np.zeros((len(bench1_values), len(bench2_values)))

    # Fill the 2D array with the bench1_fid values from the data
    for row in data:
        bench1_index = bench1_values.index(row[0])
        bench2_index = bench2_values.index(row[3])
        fid_values[bench1_index, bench2_index] = float(row[2])

    #pdb.set_trace()
    for row in data:
        bench1_index = bench1_values.index(row[0])
        bench2_index = bench2_values.index(row[3])
        fid_values[bench2_index, bench1_index] = float(row[5])

    get_avgs = [sum(i)/len(bench1_values) for i in fid_values]
    print(get_avgs)
    print(fid_values)

    print(bench1_values)
    indices_order = np.argsort(get_avgs)
    print(indices_order)
    bench1_values = np.array(bench1_values)[indices_order]
    bench2_values = np.array(bench2_values)[indices_order]

    fid_values = fid_values[indices_order]
    print(bench1_values)
    print(fid_values)

    for i in range(len(fid_values)):
        fid_values[i] = fid_values[i][indices_order]

    print(fid_values)
    # Only plot the lower triangle of the heatmap
    #mask = np.triu(len(bench1_values)*[len(bench2_values)*[1]], k=1)

    fig, ax = plt.subplots()

    #triangle = np.ma.array(bench1_fid_values, mask=mask) # mask out the lower triangle
    #cmap = plt.cm.get_cmap('jet', 10) # jet doesn't have white color
    #cmap.set_bad('w') # default value is 'k'

    fid_values = fid_values[::-1]

    #im = ax.imshow(triangle, cmap='RdYlGn')
    im = ax.imshow(fid_values, cmap='RdYlGn')

    # Set the x and y axis labels and tick labels
    ax.set_xticks(np.arange(len(bench2_values)))
    ax.set_yticks(np.arange(len(bench1_values)))
    ax.set_yticklabels(reversed(bench1_values))
    ax.set_xticklabels(bench2_values)

    # Rotate the x axis tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add a colorbar to the heatmap
    cbar = ax.figure.colorbar(im, ax=ax)
    # Set the title of the heatmap
    ax.set_title("Average fidelity values")
    plt.rcParams['savefig.dpi'] = 256
    plt.tight_layout()
    # Save the heatmap to a file
    plt.savefig("results/heatmap.png")
    file.close()
