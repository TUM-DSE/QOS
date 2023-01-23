import sys
#sys.path.insert(0, "$(pwd)")

from qiskit import IBMQ
from benchmarks import *
from backends import IBMQPU
from qiskit.compiler import transpile
#from qiskit.circuit import QuantumCircuit

class App:
    benchmark = ''
    args = []
    backend = None
    nshots = 0
    filename = ''
    filepath = ''
    provider = None
    
    def __init__(self, backend, benchmark, args, filename, filepath='', shots=1024):
            print(backend, benchmark, args, filename)
            self.provider = IBMQ.load_account()
            self.backend = IBMQPU(backend, self.provider)
            
            self.benchmark = eval(benchmark)(int(args))
            self.nshots = shots
            self.filename = filename
            self.filepath = filepath
            self.nshots = shots
            
    
    def Run(self):
        circuit = self.benchmark.circuit()
        
        qc = transpile(circuit, self.backend.backend)

        job = self.backend.backend.run(qc, self.nshots)

        counts = job.result().get_counts()
        plot_histogram(counts, filename=self.filename)
                

app = App(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
app.Run()