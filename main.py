#!/usr/bin/python3

import sys
import argparse

from qiskit import IBMQ
from benchmarks import *
from backends import IBMQPU
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram, plot_circuit_layout, plot_coupling_map
from collections import Counter

# from qiskit.circuit import QuantumCircuit


class App:
    benchmark = ""
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
        parser.add_argument("-benchmark", nargs="+")
        parser.add_argument("-bits", type=int)
        parser.add_argument("-runs", type=int)
        parser.add_argument("-shots", type=int)
        parser.add_argument("-path", required=False, default="results/")
        parser.add_argument("-rounds", type=int, required=False)
        args = parser.parse_args()

        print(args.bits)
        print(args.benchmark)
        print(args.backend)
        print(args.runs)
        print(args.path)

        exit(0)

        self.nqbits = args.bits
        self.rounds = args.rounds
        self.backend = args.backend
        self.nruns = args.runs
        self.filepath = args.path
        self.nshots = args.shots
        self.bench_args = (
            [self.nqbits] if self.rounds == None else [self.nqbits, self.rounds]
        )
        print(*self.bench_args)

        self.benchmark = eval(args.benchmark)(*self.bench_args)
        # args.benchmark is a list of the benchmarks inputted as "-benchmark GHZBenchmark HamiltonianSimulationBenchmark" for example
        # now we just call the merge function?

        self.provider = IBMQ.load_account()
        self.backend = IBMQPU(args.backend, self.provider)

        self.filename = (
            self.backend.backend.name
            + self.benchmark.name()
            + str(self.nqbits)
            + "Shots"
            + str(self.nshots)
        )

    def Run(self):
        circuit = self.benchmark.circuit()
        backend = self.backend.backend
        nqbits = self.backend.backend.num_qubits
        utilization = self.nqbits / nqbits

        qc = transpile(circuit, backend)
        avg_fid = 0

        f = open(self.filepath + self.filename + ".txt", "a")
        for i in range(self.nruns):

            if self.backend.is_simulator:
                job = backend.run(run_input=qc, shots=self.nshots)
            else:
                job = backend.run(circuits=qc, shots=self.nshots)

            counts = job.result().get_counts()
            plot_histogram(counts, filename=self.filepath + self.filename)
            plot_circuit_layout(qc, self.backend.backend)
            avg_fid = avg_fid + self.benchmark.score(Counter(counts))

            # f.write(str(counts) + "\n")
        avg_fid = avg_fid / self.nruns
        f.write(str(avg_fid))
        f.write(str(utilization))
        f.close()


app = App(sys.argv)
app.Run()
