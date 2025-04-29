# QAOA tools 
import scipy
import scipy.optimize as opt
from typing import List, Mapping, Tuple
from collections import Counter
import numpy as np
import networkx as nx
from collections import Counter
import copy  
from FrozenQubits.helper import *

from qiskit import QuantumCircuit
from qiskit_ibm_provider import IBMProvider
from qiskit.circuit import Parameter
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator


def _get_ideal_counts(circuit: QuantumCircuit) -> Counter:
    ideal_counts = {}

    if circuit.num_qubits <16:
        ideal_counts = AerSimulator().run(circuit, shots=20000).result().get_counts()
        
    else:
        provider = IBMProvider(instance="ibm-q/open/main")
        backend = provider.get_backend("simulator_statevector")
        
        ideal_counts = (
            backend.run(circuit, shots=20000).result().get_counts()
        )
        for k, v in ideal_counts.items():
            ideal_counts[k] = v / 20000

    return Counter(ideal_counts)

 
def get_benchmark_id(
        benchmark_type, 
        n,                             # number of variables (or node in graph)
        k=3,                         # k for k-regular graphs or m for BA
        T_repeat=1,              # different instance of the same problem 
        verbosity=2):       
    
    if benchmark_type.lower() == 'sk':        
        filename=f'{n-1}_{n}_{T_repeat}'
    
    elif benchmark_type.lower() == 'k_regular':                           
        if ((n*k) % 2 != 0) or (n<=k):                                                              
            if verbosity>0:                
                print(f'Invalid n={n} and k={k} inputs!')
                print('\tFor k-Regular graphs, 1<k<n and n*k must be even.')
            return None                                
        filename=f'{k}_{n}_{T_repeat}'
        
    elif benchmark_type.lower() == 'ba':
        filename=f'{k}_{n}_{T_repeat}'                        
    
    else: 
        print('Invalid benchmark type')
        return None 
    
    return filename


def pqc_QAOA(J, h=None, num_layers=1,
            add_measurement=True,  
            beta_label='b', gamma_label='g', 
            barrier_level=0):
    # print('__h', h.keys())
    weighted_edges=[(i, j, J[i,j]) for i,j in J.keys()]
    G=nx.Graph()
    if h is not None:
        G.add_nodes_from(list(h.keys()))
    G.add_weighted_edges_from(weighted_edges)    
    n=len(G.nodes())
    # print('n=', n)            
    
    params={}
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    if barrier_level>1:
        qc.barrier()    

    for p in range(1, num_layers+1):
        _g_label=f'{gamma_label}_{p}'            
        gamma = copy.deepcopy(Parameter(_g_label))
        params[_g_label]=gamma
        _b_label=f'{beta_label}_{p}'
        beta = copy.deepcopy(Parameter(_b_label))
        params[_b_label]=beta

        if h is not None:
            for i in h.keys():
                if h[i]>0 or h[i]<0:
                    qc.rz(2*h[i]*gamma, i)
        if barrier_level>1:
            qc.barrier()    

        for i,j in J.keys():
            if J[i,j]>0 or J[i,j]<0:
                qc.cx(i, j)                         
                qc.rz(2*J[i,j]*gamma, j)
                qc.cx(i, j)        
        if barrier_level>1:
            qc.barrier()    

        #Mixer Hamiltonian
        for i in range(n):
            qc.rx(2*beta,i)
        
        if barrier_level>0:
            qc.barrier()    

    if add_measurement:
        qc.measure(range(n), range(n))

    out={
        'qc':qc,
        'params':params,
        'beta_label':beta_label,
        'gamma_label':gamma_label,        
    }
    return out        

def _gen_ansatz(hamiltonian: dict[int, float], gamma: float, beta: float) -> QuantumCircuit:
    qc = QuantumCircuit(len(hamiltonian.keys()))

    # initialize |++++>
    qc.h(qubit=qc.qubits)

    # Apply the phase separator unitary
    for k, v in hamiltonian.items():
        i, j = k
        weight = v
        phi = gamma * weight

        # Perform a ZZ interaction
        qc.cnot(i, j)
        qc.rz(2 * phi, qubit=j)
        qc.cnot(i, j)

    # Apply the mixing unitary
    qc.rx(2 * beta, qubit=qc.qubits)

    qc.measure_all()

    return qc

def _get_opt_angles(pqc: QuantumCircuit, hamiltonian = dict) -> Tuple[List, float]:
    def f(params: List, pqc: QuantumCircuit, hamiltonian = dict) -> float:
        gamma, beta = params
        #circ = _gen_ansatz(hamiltonian, gamma, beta)
        circ = bind_QAOA(pqc, {'g_1': pqc.parameters[1], 'b_1': pqc.parameters[0]}, beta, gamma)
        probs = _get_ideal_counts(circ)
        objective_value = _get_expectation_value_from_probs(hamiltonian, probs)

        return -objective_value  # because we are minimizing instead of maximizing

    init_params = [np.random.uniform() * 2 * np.pi, np.random.uniform() * 2 * np.pi]
    out = opt.minimize(f, init_params, args=(pqc,hamiltonian), method="COBYLA")

    return out["x"], out["fun"]

def _gen_angles(pqc: QuantumCircuit, hamiltonian = dict) -> List:
    # Classically simulate the variational optimization 5 times,
    # return the parameters from the best performing simulation
    best_params, best_cost = [], 10.0
    for _ in range(10):
        params, cost = _get_opt_angles(pqc, hamiltonian)
        if cost < best_cost:
            best_params = params
            best_cost = cost
    return best_params

def _get_energy_for_bitstring(hamiltonian, bitstring: str) -> float:
    energy = 0
    for k, v in hamiltonian.items():
        i, j = k
        weight = v
        if bitstring[i] == bitstring[j]:
            energy -= weight  # if edge is UNCUT, weight counts against objective
        else:
            energy += weight  # if edge is CUT, weight counts towards objective
    return energy

def _get_expectation_value_from_probs(hamiltonian, probabilities: Counter) -> float:
    expectation_value = 0.0
    for bitstring, probability in probabilities.items():
        expectation_value += probability * _get_energy_for_bitstring(hamiltonian, bitstring)
    return expectation_value

def score(circuit: QuantumCircuit, counts: Mapping[str, float], hamiltonian = None) -> float:
    ideal_counts = _get_ideal_counts(circuit)
    total_shots = sum(counts.values())
    experimental_counts = Counter({k: v / total_shots for k, v in counts.items()})

    return hellinger_fidelity(ideal_counts, experimental_counts)

    ideal_value = _get_expectation_value_from_probs(hamiltonian, ideal_counts)
    experimental_value = _get_expectation_value_from_probs(hamiltonian, experimental_counts)

    return_val_1 = 1 - abs((ideal_value - experimental_value) / (2 * ideal_value))
    return_val_2 = 100 * abs((ideal_value - experimental_value) / ideal_value)

    return (return_val_1, return_val_2)

def bind_QAOA(primary_circuit, params, beta, gamma, 
                         beta_label='b', gamma_label='g'):
    gamma=to_iterable(gamma)
    beta=to_iterable(beta)
    new_circuit=primary_circuit.copy()    
    for p in range(len(beta)):
        mapping={}
        _beta= f'{beta_label}_{p+1}'
        mapping[params[_beta]]=beta[p]
        try:
            new_circuit=new_circuit.bind_parameters(mapping)
        except Exception as e:
            print(e)                                                
        mapping={}
        _gamma= f'{gamma_label}_{p+1}'
        mapping[params[_gamma]]=gamma[p]
        try:
            new_circuit=new_circuit.bind_parameters(mapping)    
        except:
            pass
    return new_circuit

