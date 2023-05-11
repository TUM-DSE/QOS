import numpy as np
from qiskit.providers.fake_provider import *

gates = {
    "u3" : 1,
    "u2" : 1,
    "u1" : 1,
    "cx" : 2,
    "id" : 1,
    "u0" : 1,
    "u" : 1,
    "p" : 1,
    "x" : 1,
    "y" : 1,
    "z" : 1,
    "h" : 1,
    "s" : 1,
    "sdg" : 1,
    "t" : 1,
    "tdg" : 1,
    "rx" : 1,
    "ry" : 1,
    "rz" : 1,
    "sx" : 1,
    "sxdg" : 1,
    "cz" : 1,
    "cy" : 1,
    "swap" : 1,
    "ch" : 1,
    "ccx" : 1,
    "cswap" : 1,
    "crx" : 1,
    "cry" : 1,
    "crz" : 1,
    "cu1" : 1,
    "cp" : 1,
    "cu3" : 1,
    "csx" : 1,
    "cu" : 1,
    "rxx" : 1,
    "rzz" : 1,
    "rccx" : 1,
    "rc3x" : 1,
    "c3x" : 1,
    "c3sqrtx" : 1,
    "c4x" : 1
}

def getGateError(backend, gate):
    cmap: list[list[int]] = backend.configuration().coupling_map
    qubits = [i for i in range(backend.configuration().n_qubits)]
    
    noiseProperties = backend.properties()
    #somelist = [x for x in cmap if noiseProperties.gate_error("cx", x) < 1]

    res = []
    
    if gates[gate] == 1:    
        for q in qubits:
            res.append(noiseProperties.gate_error(gate, q))
    else:
        for pair in cmap:
            res.append(noiseProperties.gate_error(gate, pair))

    return np.median(res)
    
def getGatesErrors(backend, dict):
    basis_gates = backend.configuration().basis_gates
    
    for g in basis_gates:
        if g == 'reset':
            continue
        try:
            dict[g + "_error"] = getGateError(backend, g)
        except:
            dict[g + "_error"] = 0
        
    return dict
   
def getGateLength(backend, gate):
    cmap: list[list[int]] = backend.configuration().coupling_map
    qubits = [i for i in range(backend.configuration().n_qubits)]
    
    noiseProperties = backend.properties()
    
    res = []
    
    if gates[gate] == 1:    
        for q in qubits:
            res.append(noiseProperties.gate_length(gate, q))
    else:
        for pair in cmap:
            res.append(noiseProperties.gate_length(gate, pair))

    return np.median(res)

def getGatesLengths(backend, dict):
    basis_gates = backend.configuration().basis_gates
    
    for g in basis_gates:
        if g == 'reset':
            continue
        try:
            dict[g + "_length"] = getGateLength(backend, g)
        except:
            dict[g + "_length"] = 0
        
    return dict

def getReadoutError(backend, dict):
    qubits = [i for i in range(backend.configuration().n_qubits)]
    noiseProperties = backend.properties()
    
    res = []
    
    for q in qubits:
        try:
            res.append(noiseProperties.readout_error(q))
        except:
            res.append(0)
        
    dict['readout_error'] = np.median(res)
    
    return dict
    
def getT1(backend):
    qubits = [i for i in range(backend.configuration().n_qubits)]
    noiseProperties = backend.properties()
    
    res = []
    
    for q in qubits:
        try:
            res.append(noiseProperties.t1(q))
        except:
            res.append(0)
        
    return np.median(res)
    
def getT2(backend):
    qubits = [i for i in range(backend.configuration().n_qubits)]
    noiseProperties = backend.properties()
    
    res = []
    
    for q in qubits:
        try:
            res.append(noiseProperties.t2(q))
        except:
            res.append(0)
        
    return np.median(res)

def getT1T2(backend, dict):
    dict['T1'] = getT1(backend)
    dict['T2'] = getT2(backend)
    
    return dict

def getBackendData(backend):
    dict = {}
    
    dict = getGatesErrors(backend, dict)
    dict = getT1T2(backend, dict)
    dict = getReadoutError(backend, dict)
    dict = getGatesLengths(backend, dict)

    return dict
        
backends = FakeProvider().backends()

for b in backends:
    print(getBackendData(b))

