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
	toReturn = QuantumCircuit(q1.num_qubits + q2.num_qubits, q1.num_clbits + q2.num_clbits)
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
			newKey = key[i * kl: i * kl + kl]

			dict.update({newKey : 0})
			for (key2, value2) in counts.items():
				if newKey == key2[i * kl: i * kl + kl]:
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
		parser.add_argument("-bits", type=int)
		parser.add_argument("-runs", type=int)
		parser.add_argument("-shots", type=int)
		parser.add_argument("-path", required=False, default="results/")
		parser.add_argument("-rounds", type=int, required=False)
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
		self.bench_args = (
			[self.nqbits] if self.rounds == None else [self.nqbits, self.rounds]
		)
		print(*self.bench_args)
		
		for b in args.benchmarks:
			self.benchmarks.append(eval(b)(*self.bench_args))
		
		self.nbenchmarks = len(self.benchmarks)
		#self.benchmark = eval(args.benchmark)(*self.bench_args)
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

	def Run(self):
		circuits = []
		
		for b in self.benchmarks:
			circuits.append(b.circuit())
		
		qc = circuits[0]
		ncircs = len(circuits)
		
		if ncircs > 1:
			qc = merge_circs(circuits[0], circuits[1])
			
			for i in range(2, ncircs):
				qc = merge_circs(qc, circuits[i])

		backend = self.backend.backend
		nqbits = self.backend.backend.num_qubits
		utilization = (self.nqbits * self.nbenchmarks) / nqbits

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
				plot_histogram(splitted_counts[i], filename=self.filepath + self.filename + str(i))
			#self.backend.backend.coupling_map.draw()
			#print(len(self.backend.backend.coupling_map))
			#plot_circuit_layout(qc, self.backend.backend)
			for i in range(self.nbenchmarks):
				avg_fid = avg_fid + self.benchmarks[i].score(Counter(splitted_counts[i]))

			# f.write(str(counts) + "\n")
		
		f = open(self.filepath + self.filename + ".txt", "a")
		avg_fid = avg_fid / (self.nbenchmarks * self.nruns)
		f.write(str(avg_fid))
		f.write('\t')
		f.write(str(utilization))
		f.close()


app = App(sys.argv)
app.Run()
