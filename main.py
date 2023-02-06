import sys
import argparse
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


def split_counts(counts, nbenchmarks):
    kl = len(list(counts.keys())[0])
    kl = int(kl / nbenchmarks)

    counts_list = []

    for i in range(0, nbenchmarks):
        dict = {}

        for (key, value) in counts.items():
            newKey = key[i * kl : i * kl + kl]

            dict.update({newKey: 0})
            for (key2, value2) in counts.items():
                if newKey == key2[i * kl : i * kl + kl]:
                    dict[newKey] = dict[newKey] + value2
        counts_list.append(dict)

    return counts_list


class App:
    benchmarks = []
    nbenchmarks = 0
    args = []
    backend: IBMQ = None
    nshots = 0
    filename = ""
    filepath = ""
    provider = None
    nruns = 0
    nbits = 0
    rounds = 0
    bench_args = ""

    # def __init__(self, backend, benchmark, nbits, nruns, filepath='', shots=1024):
    def __init__(self, *kwargs):

        config = self.config_parser()
        # pprint.pprint(data)

        print(config.benchmarks[0].name)

        self.nbits = config.benchmarks[0].nbits
        self.rounds = config.benchmarks[0].rounds
        self.nruns = config.benchmarks[0].runs
        self.filepath = config.path
        self.nshots = config.benchmarks[0].shots
        self.cuts = config.benchmarks[0].cuts
        self.backend = config.benchmarks[0].frags[0].backend

        if self.cuts != None and len(config.benchmarks) > 1:
            print(
                "Working with more than 1 benchmarks and cutting is not yet implemented"
            )
            exit(1)

        self.bench_args = (
            [self.nbits] if self.rounds == None else [self.nbits, self.rounds]
        )

        for b in config.benchmarks:
            self.benchmarks.append(eval(b.name)(*self.bench_args))

        self.nbenchmarks = len(self.benchmarks)
        # self.benchmark = eval(args.benchmark)(*self.bench_args)
        # args.benchmark is a list of the benchmarks inputted as "-benchmark GHZBenchmark HamiltonianSimulationBenchmark" for example
        # now we just call the merge function?

        self.provider = IBMQ.load_account()
        self.backend = IBMQPU(self.backend, self.provider)

        benchmark_names = ""
        for b in self.benchmarks:
            benchmark_names = benchmark_names + b.name()

        self.filename = (
            self.backend.backend.name
            + benchmark_names
            + str(self.nbits)
            + "Shots"
            + str(self.nshots)
        )

    def config_parser(self):
        with open("config.yml", "r") as config:
            data = yaml.safe_load(config)

        data = dict2obj(data)

        for i, j in enumerate(data.config.benchmarks):
            data.config.benchmarks[i] = dict2obj(j)

            for x, y in enumerate(data.config.benchmarks[i].frags):
                data.config.benchmarks[i].frags[x] = dict2obj(y)

        return data.config

    def run(self):
        circuits = []

        for b in self.benchmarks:
            circuits.append(b.circuit())
            # Where we should cut, or maybe outside of the loop, anyway we dont support yet multiple
            # benchmarks and cutting so here or outside is the same

        prf_counts = []

        for c in circuits:
            prf_counts.append(perfect_counts(c))

        qc = circuits[0]
        ncircs = len(circuits)

        if ncircs > 1:
            qc = merge_circs(circuits[0], circuits[1])

            for i in range(2, ncircs):
                qc = merge_circs(qc, circuits[i])

        backend = self.backend.backend
        nbits = self.backend.backend.num_qubits
        utilization = (self.nbits * self.nbenchmarks) / nbits

        qc = transpile(qc, backend)
        avg_fid = 0

        for i in range(self.nruns):

            if self.backend.is_simulator:
                job = backend.run(run_input=qc, shots=self.nshots)
            else:
                job = backend.run(circuits=qc, shots=self.nshots)

            counts = job.result().get_counts()
            splitted_counts = split_counts(counts, self.nbenchmarks)

            for i in range(self.nbenchmarks):
                plot_histogram(
                    splitted_counts[i], filename=self.filepath + self.filename + str(i)
                )
            # self.backend.backend.coupling_map.draw()
            # print(len(self.backend.backend.coupling_map))
            # plot_circuit_layout(qc, self.backend.backend)
            for i in range(self.nbenchmarks):
                avg_fid = avg_fid + fidelity(prf_counts[i], splitted_counts[i])

            # f.write(str(counts) + "\n")

        f = open(self.filepath + self.filename + ".txt", "a")
        avg_fid = avg_fid / (self.nbenchmarks * self.nruns)
        f.write(str(avg_fid))
        f.write("\t")
        f.write(str(utilization))
        f.close()


app = App(sys.argv)
app.run()
