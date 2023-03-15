import csv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
import pdb
import sys

def specific_bench_qbits(bench:str, qbits:int) -> int:
    
    if bench == "GHZ":
        return qbits
    elif bench == "BitCode":
        return qbits//2
    elif bench == "PhaseCode":
        return qbits//2
    elif bench == "MerminBell":
        return qbits
    elif bench == "FermionicQAOA":
        return qbits
    elif bench == "VQE":
        return qbits//2
    elif bench == "QAOA":
        return qbits
    elif bench == "Hamiltonian":
        return qbits
    else:
        return 0

# Creates an output heapmap of the static transpilation cnot difference for each benchmark and qubit count
# For this the csv has to be in the following format: "bench_name,bench_qbits,depth_before,depth_after,cnot_before,cnot_after,utilization,backend,config_file"
# To call the static plot it expects the following command line arguments: "static" qubit_count_1,qubit_count_2,...,qubit_count_n
# Where the qubits counts are the different counts that will be plotted on the x-axis, and every benchmark needs to have run for each qubit count, otherwise it will fail
if sys.argv[1] == "static":

    # Load data from CSV into a numpy array
    with open('results/results.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [row for row in reader]


    # Get unique benchmark names and qubit counts
    bench_names = np.unique([row['bench_name'] for row in data])
    #qubit_counts = np.unique([row['bench_qbits'] for row in data])
    
    #Create a numpy array from argv 2
    qubit_counts = np.array(sys.argv[2].split(','), dtype=int)

    # Create a matrix to hold the heatmap data
    heatmap_data = np.zeros((len(qubit_counts), len(bench_names)))

    # Loop over the data and populate the heatmap matrix
    for i, bench_name in enumerate(bench_names):
        for j, qubit_count in enumerate(qubit_counts):
            # Get the subset of data for the current benchmark and qubit count
            subset = [row for row in data if row['bench_name'] == bench_name and int(row['bench_qbits']) == qubit_count]
            # Compute the difference between cnot_before and cnot_after
            cnot_diff = np.mean([int(row['cnot_after']) - int(row['cnot_before']) for row in subset])
            # Store the average cnot difference in the heatmap matrix
            heatmap_data[j, i] = cnot_diff

    # Set up the heatmap figure
    fig, ax = plt.subplots(figsize=(8,6))

    pdb.set_trace()

    #Swap the bench_name MerminBell to the end of the list
    bench_names = np.delete(bench_names, bench_names.tolist().index('MerminBell'))
    bench_names = np.append(bench_names, 'MerminBell')

    tmp = heatmap_data[:, 4]
    heatmap_data = np.concatenate((heatmap_data[:,0:4], heatmap_data[:,5:]), axis=1)
    heatmap_data = np.hstack((heatmap_data, tmp.reshape(-1,1)))

    #Reverse the order of the qubit counts
    qubit_counts = qubit_counts[::-1]

    heatmap_data = heatmap_data[::-1]
    
    heatmap_data =np.ma.masked_where(heatmap_data<0, heatmap_data)

    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='blue')

    # Plot the heatmap
    im = ax.imshow(heatmap_data, cmap='RdYlGn_r')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set axis labels and tick labels
    ax.set_xticks(np.arange(len(bench_names)))
    ax.set_yticks(np.arange(len(qubit_counts)))
    ax.set_xticklabels(bench_names)
    ax.set_yticklabels(qubit_counts)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Rotate x-axis tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(len(qubit_counts)):
        for j in range(len(bench_names)):
            text = ax.text(j, i, "{:.1f}".format(heatmap_data[i, j]),
                           ha="center", va="center", color="w")

    # Set plot title and axis labels
    ax.set_title("CNOT Difference Before and After Transpilation")
    ax.set_xlabel("Benchmark Name")
    ax.set_ylabel("Benchmark qubit size")

    # Display the heatmap
    plt.rcParams['savefig.dpi'] = 256
    plt.tight_layout()
    # Save the heatmap to a file
    plt.savefig("results/heatmap.png")

# Creates an output triangle heapmap of the median score results of the combinations of benchmarks
# For this the csv has to be in the following format: "bench1,bench1_qbits,bench1_fid,bench2,bench2_qbits,bench2_fid,median_fid,utilization,backend,config_file"
if sys.argv[1] == "medians":

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
        fid_values[bench1_index, bench2_index] = float(row[6])

    get_avgs = [sum(i)/len(bench1_values) for i in fid_values]

    indices_order = np.argsort(get_avgs)
    bench1_values = np.array(bench1_values)[indices_order]
    bench2_values = np.array(bench2_values)[indices_order]

    fid_values = fid_values[indices_order]

    for i in range(len(fid_values)):
        fid_values[i] = fid_values[i][indices_order]

    # Only plot the lower triangle of the heatmap
    mask = np.triu(len(bench1_values)*[len(bench2_values)*[1]], k=1)

    fig, ax = plt.subplots()

    triangle = np.ma.array(fid_values, mask=mask) # mask out the lower triangle
    cmap = matplotlib.cm.get_cmap('jet', 10) # jet doesn't have white color
    cmap.set_bad('w') # default value is 'k'

    #triangle = triangle[::-1]
#    pdb.set_trace()
    im = ax.imshow(triangle, cmap='RdYlGn')
    #im = ax.imshow(fid_values, cmap='RdYlGn')

    # Set the x and y axis labels and tick labels
    ax.set_xticks(np.arange(len(bench2_values)))
    ax.set_yticks(np.arange(len(bench1_values)))
    ax.set_yticklabels(bench1_values)
    ax.set_xticklabels(bench2_values)

    # Rotate the x axis tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add a colorbar to the heatmap
    cbar = ax.figure.colorbar(im, ax=ax)
    # Set the title of the heatmap
    ax.set_title("Median score values")
    plt.rcParams['savefig.dpi'] = 256
    plt.tight_layout()
    # Save the heatmap to a file
    plt.savefig("results/heatmap.png")
    file.close()

# Creates an output complete heapmap of the independent score results of the combinations of benchmarks, the independent score is of the benchmark on the left (vertical axis)
#   A B C D
# A - - - -
# B - - 3 -
# C - - - -
# D - - - -
# In this example, the independent score of B when combined with C is 3
# For this the csv has to be in the following format: "bench1,bench1_qbits,bench1_fid,bench2,bench2_qbits,bench2_fid,median_fid,utilization,backend,config_file"
if sys.argv[1] == "independent":

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
    ax.set_title("Median  values")
    plt.rcParams['savefig.dpi'] = 256
    plt.tight_layout()
    # Save the heatmap to a file
    plt.savefig("results/heatmap.png")
    file.close()
