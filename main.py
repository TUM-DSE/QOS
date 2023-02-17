import sys
import argparse
import pdb
import yaml
import pprint
from qiskit import IBMQ
from benchmarks import *
from backends import IBMQPU
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram, plot_circuit_layout, plot_coupling_map
from collections import Counter

from qiskit.circuit import QuantumCircuit


class dict2obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [dict2obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, dict2obj(v) if isinstance(v, dict) else v)


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

        config = self.config_parser(self.config_file)
        print("Working on", self.config_file)
        # pdb.set_trace()
        # pprint.pprint(data)

        # print(config.benchmarks[0].name)

        self.nqbits = [i.nqbits for i in config.benchmarks]
        self.rounds = [i.rounds for i in config.benchmarks]
        self.filepath = config.path

        # For now the number of shots is for the overall application and not specific for each benchmark so no shot splitting is implemented
        self.nshots = config.nshots
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
        # `python main.py -backend FakeTorontoV2 -benchmarks GHZBenchmark BitCodeBenchmark -bits 2 3 -rounds 0 2 -runs 3 -shots 4000`
        self.bench_args = [
            [self.nqbits[i]]
            if self.rounds[i] == None
            else [self.nqbits[i]]
            if self.rounds[i] == 0
            else [self.nqbits[i], self.rounds[i]]
            for i in range(len(config.benchmarks))
        ]

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

        for i, b in enumerate(self.bench_args):
            if len(b) > 1:
                self.total_bits.append(self.nqbits[i] + (b[0] - 1) * b[1])
            elif isinstance(self.nqbits[i], list):
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

        prf_cnts = perfect_counts(qc)
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

        try:
            qc = transpile(qc, backend)
        except:
            print("Probably the circuit is too large for this backend. Skipping...")
            exit(0)
        #print(qc)
        depth_after = 0
        cnot_after = 0
        
        depth_after = qc.depth()
        cnot_after = getCNOTS(qc)
        
        print(depth_b4, "\t", depth_after)
        print(cnot_b4, "\t", cnot_after)
        
        if self.static:
            exit()

        avg_fids = [0] * self.nbenchmarks

        if self.backend.is_simulator:
            job = backend.run(run_input=qc, shots=self.nshots)
        else:
            job = backend.run(circuits=qc, shots=self.nshots)
        counts = job.result().get_counts()
        # splitted_counts = split_counts(counts, self.nbenchmarks)
        # print(counts)

        # plot_histogram(
        # counts, filename="results/counts" + ".png", figsize=(10, 10)
        # )
        splitted_counts = split_counts_bylist(counts, self.total_bits)

        for i, a in enumerate(self.nqbits):
            if isinstance(a, list):
                splitted_counts[i] = split_counts_bylist(splitted_counts[i], a)
        # print(splitted_counts)
        # splitted_counts = split_counts(counts, 12)

        # for idx, c in enumerate(splitted_counts):
        # print("-------------------")
        # plot_histogram(
        # c,
        # filename=self.filepath + self.filename + "_split_counts" + str(idx) + ".png",
        # figsize=(10, 10),
        # )
        # print(splitted_counts)

        for i in range(self.nbenchmarks):

            if isinstance(splitted_counts[i], list):

                plot_histogram(
                    splitted_counts[i][0],
                    filename=self.filepath + self.filename + "part0_" + str(i),
                    figsize=(10, 10),
                )
                plot_histogram(
                    splitted_counts[i][1],
                    filename=self.filepath + self.filename + "part1_" + str(i),
                    figsize=(10, 10),
                )
            else:
                plot_histogram(
                    splitted_counts[i],
                    filename=self.filepath + self.filename + str(i),
                    figsize=(10, 10),
                )
        # self.backend.backend.coupling_map.draw()
        # print(len(self.backend.backend.coupling_map))
        # plot_circuit_layout(qc, self.backend.backend)
        for i in range(self.nbenchmarks):
            # print(prf_counts[i])
            # print(splitted_counts[i])
            # avg_fids = avg_fids + fidelity(prf_counts[i], splitted_counts[i])
            if isinstance(splitted_counts[i], list):
                # tmp_fid = self.benchmarks[i].score(splitted_counts[i])
                avg_fids[i] += self.benchmarks[i].score(splitted_counts[i])
                # avg_fids[i] += tmp_fid / 2
            else:
                print(fidelity(prf_cnts, splitted_counts[i]))
                avg_fids[i] += self.benchmarks[i].score(Counter(splitted_counts[i]))
        # f.write(str(counts) + "\n")

        avg_fid = 0
        f = open("results/results.txt", "a")
        f.write("\n---------------------\n")
        f.write("\n" + self.filename)
        f.write("\nConfig_file: \t" + self.config_file)
        f.write("\nFidelity:")

        for i in range(self.nbenchmarks):
            fid = avg_fids[i]
            f.write("\n\t" + self.benchmarks[i].name() + ": \t" + str(round(fid, 4)))
            # print(fid)
            avg_fid = avg_fid + fid

        avg_fid = round(avg_fid / self.nbenchmarks, 4)
        f.write("\n\tFinal: \t" + str(avg_fid))
        f.write("\nUtilization: \t" + str(round(utilization, 3)))
        f.close()


app = App(sys.argv)
app.run()
