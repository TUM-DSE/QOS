from qos.kernel.matcher import Matcher
from qos.kernel.multiprogrammer import Multiprogrammer
from qos.kernel.scheduler import Scheduler 
import os
from qiskit.circuit.random import random_circuit
from qos.types import Qernel
import logging
import pdb
import matplotlib.pyplot as plt

CIRC_SIZE = 4
CIRC_DEPTH = 5
CIRC_NUMBER = 8
SUB_FREQ = 0.1
FID_WEIGHT = 0.7
UTIL_WEIGHT = 0

def main():
    
    try:
        os.system(f'redis-cli flushall')   
        os.system(f'python load_qpus.py')
    except FileNotFoundError:
      print(f"Error: The file does not exist.")

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=50)

    matcher = Matcher()
    qernels = []

    for i in range(0, CIRC_NUMBER, 3):
      print("Matching qernel {}...".format(str(i)))
      new_qernel_sub1 = Qernel(random_circuit(CIRC_SIZE, CIRC_DEPTH, max_operands=2, measure=True))
      new_qernel_sub2 = Qernel(random_circuit(CIRC_SIZE, CIRC_DEPTH, max_operands=2, measure=True))
      main_qernel = Qernel()
      new_qernel_sub1.id = i + 1
      new_qernel_sub2.id = i + 2
      main_qernel.add_subqernel(new_qernel_sub1)
      main_qernel.add_subqernel(new_qernel_sub2)
      #submit time in seconds times the sub_freq
      main_qernel.submit_time = i*SUB_FREQ
      main_qernel.id = i
      matcher.run(main_qernel)
      qernels.append(main_qernel)
    
    #qernel1.subqernels.append(subqernel1)
    #qernel1.subqernels.append(subqernel2)
    
    multi = Multiprogrammer()

    bundled_queue = multi.run(qernels)

    sched = Scheduler()

    dist = sched.run(bundled_queue, 'balanced', FID_WEIGHT, UTIL_WEIGHT)

    '''
    dist = [['IBMPerth', [('0', 2.123701361777778, 0.0, 0, 0.2505486832782954), ('1', 2.0916906666666666, 2.123701361777778, 2.023701361777778, 0.18994997527829482), ('2', 2.1144389404444444, 4.215392028444445, 4.015392028444444, 0.3044288248980095), ('3', 2.0921858275555554, 6.3298309688888885, 6.029830968888889, 0.19641612928015606), ('4', 2.101506503111111, 8.422016796444444, 8.022016796444444, 0.23486808817806015), ('5', 2.123410090666667, 10.523523299555555, 10.023523299555555, 0.24142237831028313), ('6', 2.0889818453333335, 12.646933390222221, 12.046933390222222, 0.16804076274721802), ('7', 2.092855751111111, 14.735915235555556, 14.035915235555557, 0.18983218611022556)], [[{'0011': 337, '1100': 909, '0010': 120, '0001': 446, '1111': 1202, '1101': 1048, '0100': 338, '1010': 283, '1000': 286, '0101': 230, '1011': 414, '0000': 126, '0111': 293, '1110': 1174, '0110': 424, '1001': 562}, {'0011': 15, '1100': 960, '0001': 210, '1111': 405, '0010': 43, '1101': 3472, '0100': 35, '1010': 223, '1000': 463, '0101': 123, '1011': 177, '0000': 59, '0111': 24, '1110': 510, '0110': 17, '1001': 1456}, {'1001': 131, '1100': 183, '0010': 1007, '1111': 86, '0001': 60, '1101': 65, '0100': 147, '1010': 3642, '1000': 924, '1011': 293, '0101': 51, '0000': 451, '0111': 90, '1110': 431, '0110': 519, '0011': 112}, {'1001': 414, '1100': 668, '0010': 968, '0001': 120, '1111': 139, '1101': 1090, '0100': 184, '1010': 1487, '1000': 906, '1011': 681, '0101': 346, '0000': 299, '0111': 96, '1110': 166, '0110': 137, '0011': 491}, {'1001': 1278, '1100': 736, '0001': 205, '1111': 809, '0010': 44, '1101': 1896, '0100': 56, '1010': 459, '1000': 498, '1011': 1204, '0101': 235, '0000': 53, '0111': 230, '1110': 293, '0110': 54, '0011': 142}, {'0011': 90, '1100': 943, '0010': 210, '1111': 684, '0001': 90, '0100': 954, '1101': 828, '1010': 236, '1000': 180, '0101': 206, '1011': 289, '0000': 368, '0111': 326, '1110': 760, '0110': 1841, '1001': 187}, {'0011': 54, '1100': 1028, '0010': 115, '1111': 46, '0001': 145, '1101': 62, '0100': 212, '1010': 375, '1000': 3997, '0101': 144, '1011': 45, '0000': 749, '0111': 61, '1110': 787, '0110': 262, '1001': 110}, {'1001': 443, '1100': 437, '0001': 183, '1111': 597, '0010': 486, '1101': 210, '0100': 188, '1010': 527, '1000': 148, '0101': 519, '1011': 789, '0000': 462, '0111': 615, '1110': 707, '0110': 882, '0011': 999}]]]]
    '''
    
    #pdb.set_trace()
    #plot dist as a bar chart and save to file

    #TypeError: only size-1 arrays can be converted to Python scalars

    #plt.bar(range(len(dist)), [i[1] for i in dist], align='center')
    #plt.xticks(range(len(dist)), [i[0] for i in dist])
    #plt.savefig('dist.png')
    #plt.show()

main()