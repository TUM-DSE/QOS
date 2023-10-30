from qos.kernel.matcher import Matcher
from qos.kernel.multiprogrammer import Multiprogrammer
from qos.kernel.scheduler import Scheduler 
from qos.distributed_transpiler.run import DistributedTranspiler
from qos.distributed_transpiler.virtualizer import GVInstatiator, GVKnitter

import os
from qiskit.circuit.random import random_circuit
from qos.types import Qernel
import logging
from evaluation.util import *
from evaluation.config import *
import pdb
import matplotlib.pyplot as plt
from qos.distributed_transpiler.analyser import BasicAnalysisPass


def main():
    
    #Loading qpus from qpus_availables.yml
    try:
        os.system(f'redis-cli flushall')   
        os.system(f'python load_qpus.py')
    except FileNotFoundError:
      print(f"Error: The file does not exist.")

    current_time = 0
    next_qernel_id = 0
    qernels = []

    #Generating random circuits
    for i in range(CIRC_NUMBER):
        current_time += random_time()
      
        #This is will generate 30 copies of a random benchmark divided with half the qubits
        new_qernels, next_qernel_id = get_random_circuit(next_qernel_id, current_time)      
      
        qernels += new_qernels

    # ----------------- #
    #Analysis
    basic_analysis_pass = BasicAnalysisPass()
    aggr_metadata = []

    for qrn in qernels:
        basic_analysis_pass.run(qrn)
        metadata = qrn.get_metadata()
        aggr_metadata.append([qrn.circuit, metadata["num_qubits"], metadata["depth"], metadata["num_nonlocal_gates"], metadata["num_measurements"]])

    # ----------------- #
    #Circuit cutting
    ready_qernels = []
    dt = DistributedTranspiler(size_to_reach=SIZE_TO_REACH, budget=BUDGET, methods=["GV", "WC", "QR", "QF"])
    gate_virtualizer = GVInstatiator()
    knitting = GVKnitter()

    for bc in benchmark_circuits:
        q = Qernel(bc)
        optimized_q = dt.run(q)
        ready_qernels.append(gate_virtualizer.run(optimized_q))
       

    #matcher = Matcher(qpus=backends)
    to_execute = []
    to_unpack = []
    counter = 0
    tmp_circuit = None

    print("merging")
    for i,q in enumerate(ready_qernels):
        sqs = q.get_subqernels()
        for sq in sqs:
            if i == 0 or i == 4:
                perfect_results[i] = execute(sq.get_circuit(), simulator, shots=8192)
            ssqs = sq.get_subqernels()
            for qq in ssqs:
                qc_small = qq.get_circuit()
                if tmp_circuit == None:
                    tmp_circuit = qc_small
                    to_unpack.append([])
                    to_unpack[counter].append(tmp_circuit.num_qubits)
                else:
                    if tmp_circuit.num_qubits + qc_small.num_qubits <= backend.num_qubits:
                        tmp_circuit = merge_circs(tmp_circuit, qc_small)
                        to_unpack[counter].append(qc_small.num_qubits)
                    else:
                        to_execute.append(tmp_circuit)
                        tmp_circuit = qc_small
                        to_unpack.append([])
                        counter = counter + 1
                        to_unpack[counter].append(tmp_circuit.num_qubits)
                        
                #cqc_small = transpile(qc_small, backend, optimization_level=3)
                #to_execute.append(cqc_small)

    if (len(to_execute) < len(to_unpack)):
        to_execute.append(tmp_circuit)

    print("transpiling")
    to_execute_copy = []
    for c in to_execute:
        to_execute_copy.append(transpile(c, backend, optimization_level=3))
    
    for q in ready_qernels:
        for vsq in q.get_virtual_subqernels():
            vsq.edit_metadata({"shots": 8192})

    fids = []
    for i in range(len(ready_qernels)):
        fids.append([])
       
    print("executing")
    results = execute(to_execute, backend, shots=8192, save_id=False)

    unpacked_results = []

    print("unpacking")
    for i,r in enumerate(results):
        lists = split_counts_bylist(r, to_unpack[i])
        for l in lists:
            unpacked_results.append(l)


    print("knitting")
    counter = 0
    for q in ready_qernels:
        for sq in q.get_subqernels():
            ssqs = sq.get_subqernels()
            for qq in ssqs:
                qq.set_results(unpacked_results[counter])
                counter = counter + 1

        knitting.run(q)

    for i,q in enumerate(ready_qernels):
        vsqs = q.get_virtual_subqernels()
        if i == 0 or i == 4:
            fids.append(fidelity(vsqs[3].get_results(), perfect_results[i]))
        else:
            fids.append(fidelity(vsqs[0].get_results(), perfect_results[i]))

    for i in range(len(ready_qernels)):
        aggr_metadata[i].append(np.median(fids[i]))
        aggr_metadata[i].append(np.std(fids[i]))

    write_to_csv("cut_circuits_MP_" + str(lower_limit) + ".csv", aggr_metadata)























































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

    pdb.set_trace()

    plt.bar(range(len(dist)), [i[1] for i in dist], align='center')
    plt.xticks(range(len(dist)), [i[0] for i in dist])
    plt.savefig('dist.png')
    plt.show()

main()


def execute(circuits: list[QuantumCircuit], backend: BackendV2, shots: int = 20000, save_id = False) -> list[dict[str, int]]:
    job = backend.run(circuits, shots=shots)
    
    job_id = job.job_id()
    

    if save_id:
        print(job_id)
        with open(job_id, 'w') as file:
            file.write(job_id)

    results = job.result()
    print(results.time_taken)

    return results.get_counts()