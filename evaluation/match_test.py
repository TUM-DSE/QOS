from qos.kernel.matcher import Matcher
from qos.kernel.multiprogrammer import Multiprogrammer
from qos.kernel.scheduler import Scheduler 
import os
from qiskit.circuit.random import random_circuit
from qos.types import Qernel
import logging
import pdb
import matplotlib.pyplot as plt

def main():
    
    try:
        os.system(f'redis-cli flushall')   
        os.system(f'python load_qpus.py')
    except FileNotFoundError:
      print(f"Error: The file does not exist.")

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=50)

    matcher = Matcher()
    
    qernel1 = Qernel()
    qernel1.id = "0"
    
    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subqernel1 = Qernel(qc)
    subqernel1.id = "1"

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subqernel2 = Qernel(qc)
    subqernel2.id = "2"

    qernel1.subqernels.append(subqernel1)
    qernel1.subqernels.append(subqernel2)

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subsubqernel1 = Qernel(qc)
    subsubqernel1.id = "3"

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subsubqernel2 = Qernel(qc)
    subsubqernel2.id = "4"

    subqernel2.subqernels.append(subsubqernel1)
    subqernel2.subqernels.append(subsubqernel2)

    matcher.run(qernel1)
    
    qernel2= Qernel()
    qernel2.id = "5"
    
    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subqernel1 = Qernel(qc)
    subqernel1.id = "6"

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subqernel2 = Qernel(qc)
    subqernel2.id = "7"

    qernel2.subqernels.append(subqernel1)
    qernel2.subqernels.append(subqernel2)

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subsubqernel1 = Qernel(qc)
    subsubqernel1.id = "8"

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subsubqernel2 = Qernel(qc)
    subsubqernel2.id = "9"

    subqernel2.subqernels.append(subsubqernel1)
    subqernel2.subqernels.append(subsubqernel2)

    matcher.run(qernel2)
    
    qernel3 = Qernel()
    qernel3.id = "10"
    
    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subqernel1 = Qernel(qc)
    subqernel1.id = "11"

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subqernel2 = Qernel(qc)
    subqernel2.id = "12"

    qernel3.subqernels.append(subqernel1)
    qernel3.subqernels.append(subqernel2)

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subsubqernel1 = Qernel(qc)
    subsubqernel1.id = "13"

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    subsubqernel2 = Qernel(qc)
    subsubqernel2.id = "14"

    subqernel2.subqernels.append(subsubqernel1)
    subqernel2.subqernels.append(subsubqernel2)

    matcher.run(qernel3)

    qc = random_circuit(5, 5, max_operands=2, measure=True)
    qernel4 = Qernel(qc)
    qernel4.id = "15"

    matcher.run(qernel4)
    
    multi = Multiprogrammer()

    bundled_queue = multi.run([qernel1, qernel2, qernel3, qernel4])

    sched = Scheduler()

    dist = sched.run(bundled_queue)

    #pdb.set_trace()
    #plot dist as a bar chart and save to file
    plt.bar(range(len(dist)), [i[1] for i in dist], align='center')
    plt.xticks(range(len(dist)), [i[0] for i in dist])
    plt.savefig('dist.png')
    plt.show()

main()