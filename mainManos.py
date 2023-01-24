import sys
#import argparse

from qiskit import IBMQ
from benchmarks import *
from backends import IBMQPU
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from collections import Counter
#from qiskit.circuit import QuantumCircuit

class App:
    benchmark = ''
    args = []
    backend = None
    nshots = 0
    filename = ''
    filepath = ''
    provider = None
    nruns = 0
    nqbits = 0
    
    #def __init__(self, backend, benchmark, nqbits, nruns, filepath='', shots=1024):
    def __init__(self, **kwargs):
            #print(backend, benchmark, args, filename)
            for k,v in kwargs.items():
                if "bits" in k:
                    self.nqbits = int(v)
            for k,v in kwargs.items():
                if "bench" in k:
                    self.benchmark = eval(v)(self.nqbits)
                elif "backend" in k:                    
                    self.provider = IBMQ.load_account()
                    self.backend = IBMQPU(v, self.provider)
                elif "shots" in k:
                    self.nshots = int(v)
                elif "runs" in k:
                    self.nruns = int(v)
                elif "path" in k:
                    self.filepath = v
                else:
                    raise RuntimeError("Invalid command passed")
            
            self.filename = self.backend.backend.name + self.benchmark.name() + str(nqbits) + "Shots" + str(shots)
  
    def Run(self):
        circuit = self.benchmark.circuit()
        backend = self.backend.backend
        
        qc = transpile(circuit, backend)
        avg_fid = 0
        
        for i in range(self.nruns):            
        
            if self.backend.is_simulator:
                job = backend.run(run_input=qc, shots=self.nshots)
            else:
                job = backend.run(circuits=qc, shots=self.nshots)
            
            counts = job.result().get_counts()
            plot_histogram(counts, filename = self.filepath + self.filename)
            avg_fid = avg_fid + self.benchmark.score(Counter(counts))
            
        avg_fid = avg_fid / self.nruns
        f = open(self.filepath + self.filename + ".txt", "a")
        f.write(str(avg_fid))
        f.close()
                


app = App(sys.argv)
app.Run()