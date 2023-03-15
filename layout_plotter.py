import sys
import argparse
import pdb
import yaml
import pprint
from qiskit import IBMQ
from benchmarks import *
from backends import IBMQPU
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram, plot_circuit_layout, plot_coupling_map, plot_error_map
from collections import Counter
from qiskit.transpiler import Layout
import csv
import os.path
import matplotlib.pyplot as plt
import matplotlib

from qiskit.circuit import QuantumCircuit

class dict2obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [dict2obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, dict2obj(v) if isinstance(v, dict) else v)
    
def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


def merge_circs(q1: QuantumCircuit, q2: QuantumCircuit) -> QuantumCircuit:
    toReturn = QuantumCircuit(
        q1.num_qubits + q2.num_qubits, q1.num_clbits + q2.num_clbits
    )
    qubits1 = [*range(0, q1.num_qubits)]
    clbits1 = [*range(0, q1.num_clbits)]
    qubits2 = [*range(q1.num_qubits, q1.num_qubits + q2.num_qubits)]
    clbits2 = [*range(q1.num_clbits, q1.num_clbits + q2.num_clbits)]

    toReturn.compose(q1, qubits=qubits1, clbits=clbits1, inplace=True)
    toReturn.compose(q2, qubits=qubits2, clbits=clbits2, inplace=True)

    return toReturn


def split_counts_bylist(counts, kl):
    counts_list = []
    counts_copy = {}

    for (k, v) in counts.items():
        counts_copy[k.replace(" ", "")] = counts[k]

    for i in range(len(kl)):
        dict = {}

        for (key, value) in counts_copy.items():
            newKey = key[sum(kl) - sum(kl[0 : i + 1]) : sum(kl) - sum(kl[0:i])]
            if newKey in dict:
                continue
            dict.update({newKey: 0})
            for (key2, value2) in counts_copy.items():
                if newKey == key2[sum(kl) - sum(kl[: i + 1]) : sum(kl) - sum(kl[:i])]:
                    dict[newKey] = dict[newKey] + value2

        counts_list.append(dict)

    return counts_list

def getCNOTS(c: QuantumCircuit) -> int:
    ops = c.count_ops()
    cnots = 0
    
    for (key, value) in ops.items():
            if key == 'cx':
                cnots = value
                break
    return cnots

class App:
    benchmarks = []
    nbenchmarks = 0
    args = []
    backend: IBMQ = None
    nshots = 0
    nruns = 0
    filename = ""
    filepath = ""
    provider = None
    nqbits = []
    total_bits = []
    circuits = []
    rounds = 0
    bench_args = ""
    config_file = sys.argv[1]
    static = False 

    # def __init__(self, backend, benchmark, nbits, nruns, filepath='', shots=1024):
    def __init__(self, *kwargs):

        print("Working on", self.config_file)
        config = self.config_parser(self.config_file)
        #print all the config file parameters

        # pdb.set_trace()
        # pprint.pprint(data)

        # print(config.benchmarks[0].name)

        self.nqbits = [i.nqbits for i in config.benchmarks]
        self.rounds = [i.rounds for i in config.benchmarks]
        self.nlayers = [i.nlayers for i in config.benchmarks]
        self.time_step = [i.time_step for i in config.benchmarks]
        self.total_time = [i.total_time for i in config.benchmarks]
        self.initial_state = [i.initial_state for i in config.benchmarks]
        self.filepath = config.path

        # For now the number of shots is for the overall application and not specific for each benchmark so no shot splitting is implemented
        self.nshots = config.nshots
        self.nruns = config.nruns
        # self.nshots = [i.shots for i in config.benchmarks]
        self.static = config.static
        # Cuts are also not implemented yet
        # self.cuts = [i.cuts for i in config.benchmarks]

        # For now it only supports one backend, at has to be the same for all benchmarks so I get the first one
        self.backend = config.benchmarks[0].frags[0].backend

        # if self.cuts != None and len(config.benchmarks) > 1:
        #    print(
        #        "Working with more than 1 benchmarks and cutting is not yet implemented"
        #    )
        #    exit(1)

        # If you select more that one benchmark then you have to indicate the qbits for each benchmark by the order that you
        # entered the benchmark, for example:
        # `python main.py -backend FakeTorontoV2 -benchmarks GHZBenchmark GHZBenchmark -bits 2 3 -runs 3 -shots 4000`
        # If one of the benchmarks requires to indicate the number of rounds then you need to put the same number of round's args as
        # the number of benchmarks, if the benchmark does not take rounds just put 0, for example:
        # `python main.py -backend FakeTorontoV2 -benchmarks GHZBenchmark BitCodeBenchmark -bits 2 3 -rounds 0 2 -runs 3 -shots 4000
        
        self.bench_args = []

        for i in range(len(config.benchmarks)):
            bench_args = [self.nqbits[i], self.rounds[i], self.nlayers[i], self.time_step[i], self.total_time[i], self.initial_state[i]]
            #print(bench_args)
            self.bench_args.append(list(filter(lambda x:x!=None, bench_args)))
            
        #print(self.bench_args)
        # print(self.bench_args)

        # This is a very ugly inline if else statements but it works, basically what it does is:
        # 1. If there are no rounds that the argument is None and just take the qbits argment
        # 2. If there are rounds, rounds = 0 means that the bechmark doesnt need rounds so just take the qbits
        # 3. If the rounds are different that 0 than the benchmark takes rounds that it should be on the bench_args
        # This `python main.py -backend FakeTorontoV2 -benchmarks GHZBenchmark BitCodeBenchmark -bits 2 3 -rounds 0 2 -runs 3 -shots 4000`
        # would fill the self.bench_args as [[2], [3, 2]]

        # self.bench_args = (
        #   [self.nqbits] if self.rounds == None else [self.nqbits, self.rounds]
        # )


        #If you get an error here either you are inputting more or less arguments that the benchmark needs or the initial_state does not have the
        # same number of initial values as the number of inputted qbits (Error correction)
        for idx, b in enumerate(config.benchmarks):
            self.benchmarks.append(eval(b.name)(*self.bench_args[idx]))

        self.nbenchmarks = len(self.benchmarks)

        for b in self.benchmarks:
            self.circuits.append(b.circuit())
        
        for i, c in enumerate(self.circuits):
            if isinstance(c, list):
                self.nqbits[i] = [c[0].num_qubits] * 2
            else:
                self.nqbits[i] = c.num_qubits

        #print(self.benchmarks)
        #print(self.circuits[0])
        #print(self.circuits[0][0].draw())
        #pdb.set_trace()

        for i, b in enumerate(self.bench_args):
            if self.rounds[i] != None: # This means that the benchmark is of the type ErrorCorrection.
                self.total_bits.append(self.nqbits[i] + (b[0] - 1) * b[1])
            elif isinstance(self.nqbits[i], list): #This means that this benchmark has more than one circuit which only happens on VQE
                self.total_bits.append(self.nqbits[i][0] * 2)
            else:
                self.total_bits.append(self.nqbits[i])

        # self.benchmark = eval(args.benchmark)(*self.bench_args)
        # args.benchmark is a list of the benchmarks inputted as "-benchmark GHZBenchmark HamiltonianSimulationBenchmark" for example

        # self.provider = IBMQ.load_account()
        self.backend = IBMQPU(self.backend, self.provider)

        # print(self.nqbits)
        # print(self.total_bits)

        benchmark_names = ""
        for b in self.benchmarks:
            benchmark_names = benchmark_names + b.name()

        self.filename = (
            self.backend.backend.name
            + benchmark_names
            + str(self.nqbits)
            + "Shots"
            + str(self.nshots)
        )

    def config_parser(self, config_file: str):
        with open(config_file + ".yml", "r") as config:
            data = yaml.safe_load(config)

        # Print the data dictionary
        print(data)
        print('\n')

        
        data = dict2obj(data)

        for i, j in enumerate(data.config.benchmarks):
            data.config.benchmarks[i] = dict2obj(j)

            for x, y in enumerate(data.config.benchmarks[i].frags):
                data.config.benchmarks[i].frags[x] = dict2obj(y)

        return data.config

    def run(self):
        # prf_counts = []

        # for c in circuits:
        # prf_counts.append(perfect_counts(c))

        # for idx, c in enumerate(prf_counts):
        # print("-------------------")
        # plot_histogram(
        # c,
        # filename="results/perfect_counts" + str(idx) + ".png",
        # figsize=(10, 10),
        # )
        # print(prf_counts)

        #pdb.set_trace()

        for i, a in enumerate(self.circuits):
            if isinstance(a, list):
                self.circuits[i] = merge_circs(a[0], a[1])

        qc = self.circuits[0]
        #print(qc)
        # print(qc.num_qubits)
        # print(qc.num_clbits)
        # print(qc)

        ncircs = len(self.circuits)

        if ncircs > 1:
            qc = merge_circs(self.circuits[0], self.circuits[1])

            for i in range(2, ncircs):
                qc = merge_circs(qc, self.circuits[i])

        #prf_cnts = perfect_counts(qc)
        #print(qc)
        depth_b4 = 0
        cnot_b4 = 0
        
        depth_b4 = qc.depth()
        cnot_b4 = getCNOTS(qc)
        
        backend = self.backend.backend
        # print(backend.name)
        nqbits = self.backend.backend.num_qubits
        # pdb.set_trace()

        utilization = 0
        for i in self.nqbits:
            if isinstance(i, list):
                utilization += sum(i)
            else:
                utilization += i

        utilization = utilization / nqbits

        # print(utilization)
 
        median_fids = [] * self.nbenchmarks

        for i in range(self.nbenchmarks):
            median_fids.append([0] * self.nruns)

        
        #pdb.set_trace()
 
        for k in range(self.nruns):
            try:
                qc_not_transpiled = qc
                qc = transpile(qc, backend)
            except:
                print("Probably the circuit is too large for this backend. Skipping...")
                exit(0)
            #print(qc)
            depth_after = 0
            cnot_after = 0
            
            depth_after = qc.depth()
            cnot_after = getCNOTS(qc)
            
            if self.static:
                csv_headers = []
        
                for i in range(self.nbenchmarks):
                    csv_headers.append("bench" + str(i) + "_name")
                    csv_headers.append("bench" + "_qbits")
                #append to headers the depth and cnot before and after transpilation
                csv_headers.append("depth_before")
                csv_headers.append("depth_after")
                csv_headers.append("cnot_before")
                csv_headers.append("cnot_after")
                csv_headers.append('utilization')
                csv_headers.append('backend')
                csv_headers.append('config_file')

                #write rows with the information on the headers
                file_exists = os.path.isfile('results/results.csv')

                with open("results/results.csv", mode="a", newline="") as csvfile:

                    writer = csv.writer(csvfile)
                    
                    if not file_exists:
                        writer.writerow(csv_headers)
                    
                    row = []
                    
                    for i in range(self.nbenchmarks):
                        #write benchmark name and qbits
                        row.append(self.benchmarks[i].name())
                        
                        #check if nqbits is a list, if yes just append the first element if not append the whole list
                        if isinstance(self.nqbits[i], list):
                            row.append(self.nqbits[i][0])
                        else:
                            row.append(self.nqbits[i])

                    #write depths and cnts before and after transpilation
                    #merge the following rows into a single one
                    row.append(depth_b4)
                    row.append(depth_after)
                    row.append(cnot_b4)
                    row.append(cnot_after)
                    row.append(utilization)
                    row.append(self.backend.backend.name)
                    row.append(self.config_file)
                    
                    writer.writerow(row)

                    fig, ax = plt.subplots(3, 3, figsize=(20, 15))
                    #fig, ax = plt.subplots(3, 3)

                    self.circuits[0].draw(output='mpl', ax=ax[2][0])
                    self.circuits[1].draw(output='mpl', ax=ax[2][1])
                    qc_not_transpiled.draw(output='mpl', ax=ax[2][2])

                    for i, circ in enumerate(self.circuits):
                        tmp = transpile(circ, self.backend.backend)
                        subfigure = plot_circuit_layout(tmp, self.backend.backend)
                        #subfigure.tight_layout()
                        foo = fig2rgb_array(subfigure)
                        ax[1][i].imshow(foo)
                        ax[1][i].set_axis_off()
                        ax[1][i].set_title("Independent "+ self.benchmarks[i].name()+ " with " + str(self.total_bits[i]) + " qubits mapping")

                    #this = layout_to_qubit_coordinates(qc._layout)
                    #Set figure axis to off

                    subfigure = plot_circuit_layout(qc, self.backend.backend)
                    subfigure.tight_layout()
                    foo = fig2rgb_array(subfigure)
                    ax[1][2].imshow(foo)
                    ax[1][2].set_axis_off()
                    ax[1][2].set_title("Merged mapping")
                    
                    subfigure = plot_error_map(self.backend.backend)
                    subfigure.tight_layout()
                    foo = fig2rgb_array(subfigure)
                    ax[0][1].imshow(foo)
                    ax[0][1].set_axis_off()

                    ax[0][0].set_axis_off()
                    ax[0][2].set_axis_off()

                    fig.tight_layout()

                    #Hide axis

                    #new_axs = fig.add_subplot(111)

                    #new_axs.imshow(fig_bin)

                    fig.savefig("results/circuit_layout" + ".png", dpi=1024)

                    #plot_coupling_map(num_qubits=self.backend.backend.num_qubits, qubit_coordinaties=this, filename="results/coupling_map" + ".png", figsize=(10, 10))

                    csvfile.close()

app = App(sys.argv)
app.run()
