#helper for Frozen Qubits  
import copy
import networkx as nx
import dimod
import numpy as np

def to_iterable(obj):
    ''' if obj is not an iterable (e.g., list, tuple, etc.), converts it to a tuple of size one'''
    try:
        iter(obj)
        return list(obj)
    except TypeError:
        return list([obj])




def find_hotspot(adj):
    ''' find the node with highest degree in adjacency representation of a graph 
    '''    
    if len(adj) <1:
        print('The input adjacency is empty!')
        return None
    hotspot=list(adj.keys())[0]    
    for v in adj:
        if len(list(adj[v].keys())) > len(list(adj[hotspot].keys())):
            hotspot=v
    return hotspot

def get_nodes_sorted_by_degree(adj):
    if len(adj) <1:
        print('The input adjacency is empty!')
        return None
    
    nodes = {}

    for v in adj:
        nodes[v] = len(adj[v].keys())

    nodes = dict(sorted(nodes.items(), key=lambda x:x[1], reverse=True))

    return nodes


def drop_hotspot_node(G, 
        target_node=None,
        list_of_fixed_vars=[], verbosity=1):
    _G=copy.deepcopy(G)
    if target_node is None:
        target_node=find_hotspot(_G.adj)
        if verbosity>0:
            print('Target node had not been specified!')
            print('Targetting the node with highest degree: ', target_node)
    _G.remove_node(target_node)
    list_of_fixed_vars.append(target_node)
    return _G, list_of_fixed_vars


def encode_var_names(bqm):
    _bqm=copy.deepcopy(bqm)
    old_vars=sorted(list(dict(_bqm.adj).keys()))
    n=len(old_vars)
    _encoding={old_vars[i]:i for i in range(n)}
    new_bqm=copy.deepcopy(_bqm)
    new_bqm.relabel_variables(_encoding)    
    # print('old_vars', old_vars)
    # print(_encoding)
    # new_vars=sorted(list(dict(new_bqm.adj).keys()))
    # print('new_vars', new_vars)
    # print(new_bqm)
    # print('--------------')

    return new_bqm, _encoding


def decode_var_names(encoding, J, h=None, offset=0.0):    
    _bqm=dimod.BinaryQuadraticModel.from_ising(h, J, offset)        
    _decoding={j:i for i,j in encoding.items()}
    new_bqm=copy.deepcopy(_bqm)
    new_bqm.relabel_variables(_decoding)    
    # print('old_vars', old_vars)
    # print(_encoding)
    # new_vars=sorted(list(dict(new_bqm.adj).keys()))
    # print('new_vars', new_vars)
    # print(new_bqm)
    # print('--------------')

    return new_bqm, _decoding


def halt_qubits(J, h={}, offset=0.0, halting_list=[], get_all_sub_Ising_problems=True):
    _bqm=dimod.BinaryQuadraticModel.from_ising(h, J, offset)    
    _halting_list=to_iterable(copy.deepcopy(halting_list))
    h_for_remove={i:0 for i in _halting_list}
    sampleset=dimod.ExactSolver().sample_ising(h_for_remove, {})
    
    #we need to generate 2^k (smaller) Ising problems 
    sub_Ising_list=[]
    for s in sampleset.samples():
        Ising_obj={}
        _val=copy.deepcopy(dict(s))        
        Ising_obj['halting_vals']=copy.deepcopy(_val)
        bqm_i=copy.deepcopy(_bqm)
        for q, v in _val.items():                        
            bqm_i.fix_variable(q, v)                                    
            # print(bqm_i.linear, bqm_i.quadratic, bqm_i.offset)
        bqm_i, encoding=encode_var_names(bqm_i)                    
        Ising_obj['J']=copy.deepcopy(dict(bqm_i.quadratic))
        Ising_obj['h']=copy.deepcopy(dict(bqm_i.linear))
        Ising_obj['offset']=bqm_i.offset
        Ising_obj['encoding']=copy.deepcopy(encoding)        
        sub_Ising_list.append(copy.deepcopy(Ising_obj))        

        if not get_all_sub_Ising_problems:
            print('Braking')
            break
    
    return sub_Ising_list