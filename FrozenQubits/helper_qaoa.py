# QAOA tools 

import numpy as np
import networkx as nx
from collections import Counter
import copy  
from helper import *

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


 
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
        returnNone 
    
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
        except:
            pass                                                
        mapping={}
        _gamma= f'{gamma_label}_{p+1}'
        mapping[params[_gamma]]=gamma[p]
        try:
            new_circuit=new_circuit.bind_parameters(mapping)    
        except:
            pass
    return new_circuit

