#!/usr/bin/python3

import sys
import argparse

from qiskit import IBMQ
from benchmarks import *
from backends import IBMQPU
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram, plot_circuit_layout, plot_coupling_map
from collections import Counter

from qiskit.circuit import QuantumCircuit


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
    print(kl)

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


def split_counts_bylist(counts, kl):
    counts_list = []

    for i in range(len(kl)):
        dict = {}

        for (key, value) in counts.items():
            newKey = key[sum(kl[0:i]) : sum(kl[0:i]) + kl[i]]
            dict.update({newKey: 0})
            for (key2, value2) in counts.items():
                if newKey == key2[sum(kl[0:i]) : sum(kl[0:i]) + kl[i]]:
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
    nqbits = 0
    rounds = 0
    bench_args = ""

    # def __init__(self, backend, benchmark, nqbits, nruns, filepath='', shots=1024):
    def __init__(self, *kwargs):

        parser = argparse.ArgumentParser()
        parser.add_argument("-backend")
        parser.add_argument("-benchmarks", nargs="+", type=str)
        parser.add_argument("-bits", nargs="+", type=int)
        parser.add_argument("-runs", type=int)
        parser.add_argument("-shots", type=int)
        parser.add_argument("-path", required=False, default="results/")
        parser.add_argument("-rounds", type=int, nargs="+", required=False)
        parser.add_argument("-cuts", type=int, required=False)

        args = parser.parse_args()

        # print(args.bits)
        # print(args.rounds)
        # print(args.benchmarks)
        # print(args.backend)
        # print(args.runs)
        # print(args.path)

        self.nqbits = args.bits
        self.rounds = args.rounds
        self.backend = args.backend
        self.nruns = args.runs
        self.filepath = args.path
        self.nshots = args.shots
        self.cuts = args.cuts

        if self.cuts != None and len(args.benchmarks) > 1:
            print(
                "Working with more than 1 benchmarks and cutting is not yet implemented"
            )
            exit(1)

        # If you select more that one benchmark then you have to indicate the qbits for each benchmark by the order that you
        # entered the benchmark, for example:
        # `python main.py -backend FakeTorontoV2 -benchmarks GHZBenchmark GHZBenchmark -bits 2 3 -runs 3 -shots 4000`
        # If one of the benchmarks requires to indicate the number of rounds then you need to put the same number of round's args as
        # the number of benchmarks, if the benchmark does not take rounds just put 0, for example:
        # `python main.py -backend FakeTorontoV2 -benchmarks GHZBenchmark BitCodeBenchmark -bits 2 3 -rounds 0 2 -runs 3 -shots 4000`
        self.bench_args = [
            [self.nqbits[i]]
            if self.rounds == None
            else [self.nqbits[i]]
            if self.rounds[i] == 0
            else [self.nqbits[i], self.rounds[i]]
            for i in range(len(args.benchmarks))
        ]

        print(self.bench_args)

        # This is a very ugly inline if else statements but it works, basically what it does it:
        # 1. If there are no rounds that the argument is None and just take the qbits argment
        # 2. If there are rounds, rounds = 0 means that the bechmark doesnt need rounds so just take the qbits
        # 3. If the rounds are different that 0 than the benchmark takes rounds that it should be on the bench_args
        # This `python main.py -backend FakeTorontoV2 -benchmarks GHZBenchmark BitCodeBenchmark -bits 2 3 -rounds 0 2 -runs 3 -shots 4000`
        # would fill the self.bench_args as [[2], [3, 2]]

        # self.bench_args = (
        #   [self.nqbits] if self.rounds == None else [self.nqbits, self.rounds]
        # )

        # For some reason enumerate is not working correclty here
        for idx in range(len(args.benchmarks)):
            self.benchmarks.append(eval(args.benchmarks[idx])(*self.bench_args[idx]))

        self.nbenchmarks = len(self.benchmarks)

        # self.benchmark = eval(args.benchmark)(*self.bench_args)
        # args.benchmark is a list of the benchmarks inputted as "-benchmark GHZBenchmark HamiltonianSimulationBenchmark" for example

        # self.provider = IBMQ.load_account()
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

        for b in self.benchmarks:
            print(b.circuit())
            circuits.append(b.circuit())

        prf_counts = []

        for c in circuits:
            prf_counts.append(perfect_counts(c))

        qc = circuits[0]
        ncircs = len(circuits)

        if ncircs > 1:
            qc = merge_circs(circuits[0], circuits[1])

            for i in range(2, ncircs):
                qc = merge_circs(qc, circuits[i])

        print(qc)

        backend = self.backend.backend
        print(backend.name)
        nqbits = self.backend.backend.num_qubits
        utilization = (sum(self.nqbits)) / nqbits

        qc = transpile(qc, backend)
        avg_fid = 0

        for i in range(self.nruns):

            if self.backend.is_simulator:
                job = backend.run(run_input=qc, shots=self.nshots)
            else:
                job = backend.run(circuits=qc, shots=self.nshots)

            counts = job.result().get_counts()
            splitted_counts = split_counts(counts, self.nbenchmarks)
            print(counts)
            print(splitted_counts)
            # splitted_counts = split_counts_bylist(counts, self.nqbits)
            # print(splitted_counts)

            for i in range(self.nbenchmarks):
                plot_histogram(
                    splitted_counts[i], filename=self.filepath + self.filename + str(i)
                )
            # self.backend.backend.coupling_map.draw()
            # print(len(self.backend.backend.coupling_map))
            # plot_circuit_layout(qc, self.backend.backend)

            for i in range(self.nbenchmarks):
                # print(prf_counts[i])
                # print(splitted_counts[i])
                # print(fidelity(prf_counts[i], splitted_counts[i]))
                avg_fid = avg_fid + fidelity(prf_counts[i], splitted_counts[i])

            # f.write(str(counts) + "\n")

        f = open(self.filepath + self.filename + ".txt", "a")
        avg_fid = avg_fid / (self.nbenchmarks * self.nruns)
        print(avg_fid)
        f.write(str(avg_fid))
        f.write("\t")
        f.write(str(utilization))
        f.close()


app = App(sys.argv)
app.run()
