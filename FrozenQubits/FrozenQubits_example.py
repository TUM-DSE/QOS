#!/usr/bin/env python
# coding: utf-8

# # FrozenQubits:   Boosting Fidelity of QAOA by Skipping Hotspot Nodes
# 
# FrozenQubits is a novel quantum divide-and-conquer approach that leverages the insight about most real-world applications having power-law graphs to boost the fidelity of QAOA applications. 
# In FrozenQubits, we freeze some of the nodes with the highest degree of connectivity. 
# This drops the edges connected to these nodes resulting in a circuit with fewer CNOTs and lower depth. 
# 
# To read the full paper, please visit: (https://arxiv.org/abs/2210.17037)(https://arxiv.org/abs/2210.17037)
# 
# This tutorial includes some example codes showing how to apply FrozenQubits to QAOA applications and compare its results with the standard QAOA. 
# 

# ## Requirements
# 
# The following packages are needed to run the sample codes:
# 
# - numpy (pip install numpy)
# - qiskit (pip install qiskit)
# - networkx (pip install networkx)
# - dimod (pip install dimod)

# In[1]:


from helper import *
from helper_qaoa import *
from  helper_FrozenQubits import *

import qiskit
from qiskit import *


# ## Specifying Benchmark Problem
# 
# This repository includes three types of problems:
# - SK: Sherrington—kirkpatrick (SK) model or fully-connected graphs 
# - k_regular: 3-Regular graphs
# - ba: Barabasi—Albert (BA) or power-law graphs 
# 
# 
# In addition to the benchmark type, we should specify the number of qubits (or nodes of the problem graph), denoted by $n \in \{4, 5, \dots, 24\}$ and the value of $k.$
# - In "k_regular" graphs, $k$ specifies the degree of the nodes, and only $k=3$ examples are available in this repository. 
# - In BA graphs, $k$ specifies the preferential attachment $(d_{BA}),$ and $k=1,2$ and 3 benchmarks are included in this repository.
# 

# In[2]:


benchmark_type=['sk', 'k_regular', 'ba'][2]

# Number of nodes/qubits
n=8
print(f'Benchmark: {benchmark_type} graph with {n} nodes')

k=1
# in BA graphs, k=1,2 or 3
#for k_regular, only k=3 is available 
# Also, for k-Regular graphs, 1<k<n and k*n must be an even number

# We only have data for P=1 layers of QAOA 
num_layers=1

try:
    benchmark_id=get_benchmark_id(benchmark_type, n=n, k=k)+f'^P={num_layers}'
except:
    pass
    # in k-regular graphs, 1<k<n, and n*k ust be an even number


# In[ ]:





# ## Loading Benchmark Problem

# In[3]:


# baseline files are in the following folder:
qaoa_path=f'./../experiments/qaoa/{benchmark_type}/gridsearch_100/'

# Results for the ideal simulator on in the "ideal" subfolder 
qaoa_ideal_filename=f'{qaoa_path}ideal/{benchmark_id}.pkl'
qaoa_ideal_obj=load_pickle(qaoa_ideal_filename)
#qaoa_ideal_obj is a dict object 

# all benchmark files include h, J and offset that defines the Ising Hamiltonian 
# h denotes linear coefficients
# J denotes quadratic coefficients
# offset is the constant bias
h=qaoa_ideal_obj['h']
J=qaoa_ideal_obj['J']
offset=qaoa_ideal_obj['offset']

print(f'{benchmark_type} benchmark with {n} nodes loaded')
print('h=', h)
print('J=', J)
print('offset=', offset)

# for each benchmark, we have found the optimum solution(s) by Brute force approach
cost_best=qaoa_ideal_obj['cost_best']
optimum_solutions=qaoa_ideal_obj['optimum_solutions']

print('--------')
print('Minimum cost value=', cost_best)
print('Optimum solutions:', optimum_solutions)


# ## Creating Parametric QAOA Circuit 

# In[4]:


# the "helper_qaoa.py" (which has been published with this tutorial) 
# includes "pqc_QAOA" module that creates a parametric QAOA circuit 
# for an Ising Hamiltonian, defined with h and J
_out=pqc_QAOA(J=J, h=h, num_layers=num_layers)
qaoa_circ=_out['qc']

print(f'QAOA circuit with P={num_layers} layers created')
print('Number of qubits:', qaoa_circ.num_qubits)
print('Depth:', qaoa_circ.depth())
print('CNOT count:', qaoa_circ.count_ops()['cx'])


# In[5]:


#and here is the circuit layout 
print(qaoa_circ)


# ## Training Circuit Parameters $(\gamma$ and $\beta)$

# In[6]:


# We must use a classical optimizer to find the optimum values of circuit parameters, gamma and beta 

# We already have searched on the grid of 100x100 to find the optimum parameter values
x_best=qaoa_ideal_obj['x_best']
_gamma=x_best[0]
_beta=x_best[1]
print('Optimum circuit parameters (num_layers=1):')
print('\t γ=', _gamma)
print('\t β=', _beta)

EV_best=qaoa_ideal_obj['EV_best']
print('EV_best (ideal)=', EV_best)
print('Approximation Ratio (ideal)=', EV_best/cost_best)


# ## Running the QAOA Circuit on a Real Quantum Hardware

# In[7]:


# Now, we can replace circuit parameters with optimum values of gamma and beta
# compile it to a real quantum computer,
# and run it on a real device
qaoa_circ_binded=bind_QAOA(qaoa_circ, _out['params'], gamma=_gamma, beta=_beta)                        

# we already have provided the qasm file of the circuit with optimum values
qaoa_ideal_qasm_filename=f'{qaoa_path}ideal/{benchmark_id}.qasm'
circ=qiskit.circuit.QuantumCircuit.from_qasm_file(qaoa_ideal_qasm_filename)

# For all benchmarks, we have run the circuit with optimum parameters 
# on eight different quantum computers by IBM
machine_name='ibmq_montreal'
# results from running the circuit with optimum parameter values are in the folders
# with IBM machine names
# results for 8 different IBM machines are available for all benchmarks

qaoa_real_filename=f'{qaoa_path}{machine_name}/{benchmark_id}.pkl'
qaoa_real_obj=load_pickle(qaoa_real_filename)
# qaoa_real_obj has a structure that is similar to qaoa_ideal_obj

EV_real_best=qaoa_real_obj['EV_best']
print(f'EV_best ({machine_name})=', EV_real_best)
print(f'Approximation Ratio ({machine_name}) =', EV_real_best/cost_best)


# ## Applying FrozenQubits
# 
# In FrozenQubits, we must specify the number of qubits to freeze, denoted by $m.$
# We have results for $m=1$ and 2.

# ## Identifying and Removing Hotspot nodes

# In[8]:


m=1
# number of qubits to freeze
# results for m=1 and 2 have been included in this repository 

G=nx.Graph()
G.add_edges_from(list(J.keys()))
G.add_nodes_from(list(h.keys()))
print('Edges: ', G.edges)

list_of_halting_qubits=[]
for i in range(m):
    G, list_of_halting_qubits=drop_hotspot_node(G, list_of_fixed_vars=list_of_halting_qubits, verbosity=0)                                                        

print('Nodes to drop:', list_of_halting_qubits)


# ## Freeze Qubits and Create Sub-Problems
# 
# Freezing $m$ qubits will result in $2^m$ sub-problems.
# 
# Note: Each sub-problem has its own $h$, $J$ and offset

# In[9]:


sub_Ising_list=halt_qubits(J=J, h=h, offset=offset, halting_list=list_of_halting_qubits)                

print(f'For m={m}, {len(sub_Ising_list)} sub-problems created')



# In[11]:


# exploring the sub-problems 

for sub_index in range(len(sub_Ising_list)):
    print('------Sub-problem number:', sub_index )
    sub_problem=sub_Ising_list[sub_index ]    
    print('h:', sub_problem['h'])
    print('J', sub_problem['J'])
    print('offset:', sub_problem['offset'])
    # most quantum devices do not accept arbitrary qubit numbers, 
    # so we encode node names 
    print('encoding node names:', sub_problem['encoding'])
    print('FrozenQubits values:', sub_problem['halting_vals'])
    print()


# ## Creating QAOA Circuits for Sub-Problems

# In[12]:


# For each sub-problem, we can create a QAOA circuit 
#  Note: each sub-problem includes its own h, J and offset

#Example: Creating QAOA circuit 
# for the first sub-problem
sub_problem=sub_Ising_list[0]    
_out_i=pqc_QAOA(J=sub_problem['J'], h=sub_problem['h'], num_layers=num_layers)
qc_i=_out_i['qc']

print('QAOA circuit for sub-problem created')
print('Number of Qubits:', qc_i.num_qubits)
print('Circuit Depth:', qc_i.depth())
print('CNOT Count:', qc_i.count_ops().get('cx', []))


# ## Training Sub-QAOA Circuitsof 

# In[14]:


# we have included results of training all sub-problems in this repository 

# FrozenQubits files are in the following folder:
FQ_path=f'./../experiments/frozenqubits_full/{benchmark_type}/gridsearch_100/'

sub_index=0
# sub_index can be between 0 to 2^m -1
FQ_id=get_benchmark_id(benchmark_type, n=n, k=k)+f'^M={m}_{sub_index}^P={num_layers}'

# Results for the ideal simulator is in in the "ideal" subfolder 
FQ_ideal_filename_i=f'{FQ_path}ideal/{FQ_id}.pkl'
FQ_ideal_obj_i=load_pickle(FQ_ideal_filename_i)
#FQ_ideal_obj is a dict object 

print(f'm={m} | sub-index={sub_index}')
print(f'EV_best (ideal)=', FQ_ideal_obj_i['EV_best'])
print(f'Approximation Ratio (ideal)= ', FQ_ideal_obj_i['EV_best']/FQ_ideal_obj_i['cost_best'])


# ## Running Sub-Problems on a Real Quantum Hardware

# In[17]:


# Results from running all sub-problems on eight different IBM machines are also included 

FQ_real_filename_i=f'{FQ_path}{machine_name}/{FQ_id}.pkl'
FQ_real_obj_i=load_pickle(FQ_real_filename_i)
#FQ_real_obj_iis a dict object 

print('Machine:', machine_name)
print(f'EV_best ({machine_name})=', FQ_real_obj_i['EV_best'])
print(f'Approximation Ratio ({machine_name})=', FQ_real_obj_i['EV_best']/FQ_real_obj_i['cost_best'])


# ## Comparing Results

# In[19]:


print('Benchmark Type:', benchmark_type)
print('Number of problem variables:', n)
print(f'k={k} (for SK model, k is ignored)')

print()
print('Baseline (standard QAOA)')
print(f'\t EV (ideal):', qaoa_ideal_obj['EV_best'])
print(f'\t EV ({machine_name}):', qaoa_real_obj['EV_best'])
print(f'\t AR (ideal):', qaoa_ideal_obj['EV_best']/qaoa_ideal_obj['cost_best'])
print(f'\t AR ({machine_name}):', qaoa_real_obj['EV_best']/qaoa_ideal_obj['cost_best'])

print('--------------------------------')
print(f'FrozenQubits (m={m})')
for sub_index in range(2**m):
    print(f'  sub-problem ', sub_index)
    FQ_id=get_benchmark_id(benchmark_type, n=n, k=k)+f'^M={m}_{sub_index}^P={num_layers}'
    FQ_ideal_filename_i=f'{FQ_path}ideal/{FQ_id}.pkl'
    FQ_ideal_obj_i=load_pickle(FQ_ideal_filename_i)
    FQ_real_filename_i=f'{FQ_path}{machine_name}/{FQ_id}.pkl'
    FQ_real_obj_i=load_pickle(FQ_real_filename_i)
    print(f'\t EV (ideal):', FQ_ideal_obj_i['EV_best'])
    print(f'\t EV ({machine_name}):', FQ_real_obj_i['EV_best'])
    print(f'\t AR (ideal):', FQ_ideal_obj_i['EV_best']/FQ_ideal_obj_i['cost_best'])
    print(f'\t AR ({machine_name}):', FQ_real_obj_i['EV_best']/FQ_real_obj_i['cost_best'])
    


# In[ ]:





# In[ ]:




