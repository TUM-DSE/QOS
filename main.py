#!python

import sys
import argparse
from qvm.cut import cut
from qvm.knit import merge
from qvm.bisection import Bisection
from qiskit import IBMQ
from benchmarks import *
from backends import IBMQPU
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram, plot_circuit_layout, plot_coupling_map
from typing import List
from qiskit.circuit import QuantumCircuit
from qvm.prob import ProbDistribution


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


def bisect_circuit(circuit: QuantumCircuit, cuts: int):

    if cuts < 1:
        return [circuit]

    passes = Bisection()
    distcircuit = cut(circuit, passes)

    frags = distcircuit.fragments
    frags = [distcircuit.fragment_as_circuit(frags[i]) for i in [0, 1]]

    final_frags = []
    final_frags += bisect_circuit(frags[0], cuts - 1)
    final_frags += bisect_circuit(frags[1], cuts - 1)

    return final_frags


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
    nqbits = 0
    rounds = 0
    bench_args = ""

    # def __init__(self, backend, benchmark, nqbits, nruns, filepath='', shots=1024):
    def __init__(self, *kwargs):

        parser = argparse.ArgumentParser()
        parser.add_argument("-backend")
        parser.add_argument("-benchmarks", nargs="+", type=str)
        parser.add_argument("-bits", type=int)
        parser.add_argument("-runs", type=int)
        parser.add_argument("-shots", type=int)
        parser.add_argument("-path", required=False, default="results/")
        parser.add_argument("-rounds", type=int, required=False)
        parser.add_argument("-cuts", nargs="+", type=int, required=False)

        args = parser.parse_args()

        print(args.bits)
        print(args.benchmarks)
        print(args.backend)
        print(args.runs)
        print(args.path)

        self.nqbits = args.bits
        self.rounds = args.rounds
        self.backend = args.backend
        self.nruns = args.runs
        self.filepath = args.path
        self.nshots = args.shots
        self.cuts = args.cuts

        if self.cuts != None and len(self.cuts) != len(args.benchmarks):
            print(
                "It was selected "
                + str(len(args.benchmarks))
                + " benchmarks but was only indicated "
                + str(len(self.cuts))
                + " cuts. It needs to be the same cuts as benchmarks"
            )
            exit(1)

        self.bench_args = (
            [self.nqbits] if self.rounds == None else [self.nqbits, self.rounds]
        )
        # print(*self.bench_args)

        for b in args.benchmarks:
            self.benchmarks.append(eval(b)(*self.bench_args))

        self.nbenchmarks = len(self.benchmarks)
        # self.benchmark = eval(args.benchmark)(*self.bench_args)
        # args.benchmark is a list of the benchmarks inputted as "-benchmark GHZBenchmark HamiltonianSimulationBenchmark" for example
        # now we just call the merge function?

        self.provider = IBMQ.load_account()
        self.backend = IBMQPU(args.backend, self.provider)

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

    def run(self):
        circuits = []

        for i, b in enumerate(self.benchmarks):
            # print(b.circuit())
            if self.cuts != None:
                frags = []
                frags += bisect_circuit(b.circuit(), self.cuts[i])
                circuits.append(frags)
            else:
                circuits.append(b.circuit())

            # print("Final frags ", len(frags))
            # for i in frags:
            #    print(i)

        prf_counts = []

        # print("Circuits ", len(circuits))
        # This loops through the benchmarks
        for c in self.benchmarks:
            prf_counts.append(perfect_counts(c.circuit()))

        # Now merging the fragments of the circuit and the circuits as a whole circuit

        if self.cuts != None:
            qc = circuits[0][0]

            for i in circuits:
                for j in i:
                    qc = merge_circs(qc, j)
        else:
            qc = circuits[0]

            for i in circuits:
                qc = merge_circs(qc, i)

        backend = self.backend.backend
        nqbits = self.backend.backend.num_qubits
        # utilization = (self.nqbits * self.nbenchmarks) / nqbits
        utilization = qc.num_qubits / nqbits

        print(qc)
        print(utilization)
        qc = transpile(qc, backend)
        avg_fid = 0

        final_counts = []

        for i in range(self.nruns):
            if self.backend.is_simulator:
                job = backend.run(run_input=qc, shots=self.nshots)
            else:
                job = backend.run(circuits=qc, shots=self.nshots)

            counts = job.result().get_counts()
            # print(counts)
            bench_split_counts = split_counts(counts, self.nbenchmarks)
            # In this step the split_counts splits the counts per benchmark, however each benchmark can
            # also be composed of multiple fragments which are split next

            # print(bench_split_counts)
            # print(bench_split_counts[0])
            # print(bench_split_counts[1])
        if self.cuts != None:
            for idx, j in enumerate(bench_split_counts):
                frag_counts = split_counts(j, 2 ** (self.cuts[idx]))
                # print(frag_counts)
                # print(ProbDistribution.from_counts(frag_counts[i]))
                frag_probs = [
                    ProbDistribution.from_counts(frag_counts[w])
                    for w in range(len(frag_counts))
                ]
                final_counts.append(merge(frag_probs).counts())
                # final_counts.append(merge(frag_counts).counts())

                for i in range(self.nbenchmarks):
                    plot_histogram(
                        final_counts[i],
                        filename=self.filepath + self.filename + str(i),
                    )

                for i in range(self.nbenchmarks):
                    avg_fid = avg_fid + fidelity(prf_counts[i], final_counts[i])
        else:
            for i in range(self.nbenchmarks):
                plot_histogram(
                    bench_split_counts[i],
                    filename=self.filepath + self.filename + str(i),
                )
            for i in range(self.nbenchmarks):
                avg_fid = avg_fid + fidelity(prf_counts[i], bench_split_counts[i])

            print(prf_counts)
            print(avg_fid)

        f = open(self.filepath + self.filename + ".txt", "a")
        avg_fid = avg_fid / (self.nbenchmarks * self.nruns)
        f.write(str(avg_fid))
        f.write("\t")
        f.write(str(utilization))
        f.close()


app = App(sys.argv)
app.run()
