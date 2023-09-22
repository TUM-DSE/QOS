from qos.engines.matcher import Matcher
from qos.engines.multiprogrammer import Multiprogrammer
import os
from qiskit.circuit.random import random_circuit
from qos.types import Qernel
import pdb

def main():
    
    try:    
        os.system(f'python qpus_available.py')
    except FileNotFoundError:
      print(f"Error: The file does not exist.")

    matcher = Matcher()
    
    qernel1 = Qernel()
    qernel1.provider = "0"
    
    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subqernel1 = Qernel(qc)
    subqernel1.provider = "1"

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subqernel2 = Qernel(qc)
    subqernel2.provider = "2"

    qernel1.subqernels.append(subqernel1)
    qernel1.subqernels.append(subqernel2)

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subsubqernel1 = Qernel(qc)
    subsubqernel1.provider = "3"

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subsubqernel2 = Qernel(qc)
    subsubqernel2.provider = "4"

    subqernel2.subqernels.append(subsubqernel1)
    subqernel2.subqernels.append(subsubqernel2)

    matcher.run(qernel1)
    
    qernel2= Qernel()
    qernel2.provider = "5"
    
    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subqernel1 = Qernel(qc)
    subqernel1.provider = "6"

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subqernel2 = Qernel(qc)
    subqernel2.provider = "7"

    qernel2.subqernels.append(subqernel1)
    qernel2.subqernels.append(subqernel2)

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subsubqernel1 = Qernel(qc)
    subsubqernel1.provider = "8"

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subsubqernel2 = Qernel(qc)
    subsubqernel2.provider = "9"

    subqernel2.subqernels.append(subsubqernel1)
    subqernel2.subqernels.append(subsubqernel2)

    matcher.run(qernel2)
    
    qernel3 = Qernel()
    qernel3.provider = "10"
    
    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subqernel1 = Qernel(qc)
    subqernel1.provider = "11"

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subqernel2 = Qernel(qc)
    subqernel2.provider = "12"

    qernel3.subqernels.append(subqernel1)
    qernel3.subqernels.append(subqernel2)

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subsubqernel1 = Qernel(qc)
    subsubqernel1.provider = "13"

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subsubqernel2 = Qernel(qc)
    subsubqernel2.provider = "14"

    subqernel2.subqernels.append(subsubqernel1)
    subqernel2.subqernels.append(subsubqernel2)

    matcher.run(qernel3)

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    qernel4 = Qernel(qc)
    qernel4.provider = "15"

    matcher.run(qernel4)
    
    multi = Multiprogrammer()

    #pdb.set_trace()
    bundled_queue = multi.run([qernel1, qernel2, qernel3, qernel4])

    

main()