from typing import Dict, Any, List
import yaml
#import pdb
import qos.database as db
from qos.types.types import Qernel
from qos.backends.types import QPU
import ast
from qiskit import QuantumCircuit
import logging
from qiskit.converters import circuit_to_dag
from qiskit.providers.fake_provider import *
from qiskit.compiler import transpile
from qos.time_estimator.basic_estimator import CircuitEstimator
#from settings.ibm_token import IBM_TOKEN
import pickle


def check_layout_overlap(layout1: List, layout2: List) -> bool:
    for i in range(0, len(layout1)):
        if layout1[i] in layout2:
            return True
    return False

def size_overflow(new_qernel:Qernel, queued_qernel:Qernel, match_qpu:str) -> bool:
    max_qubits = db.getQPU_fromname(match_qpu).max_qubits
    if new_qernel.circuit.num_qubits + queued_qernel.circuit.num_qubits > max_qubits:
        return True
    else:
        return False
    
def bundle_qernels(new_qernel:Qernel, queued_qernel:Qernel, queued_matching=tuple) -> None:
        
        if queued_qernel.src_qernels == []:
            bundled_qernel = Qernel(queued_qernel.circuit)
            bundled_qernel.append_circuit(new_qernel.circuit)
            bundled_qernel.src_qernels.append(queued_qernel)
            bundled_qernel.src_qernels.append(new_qernel)
            bundled_qernel.match = queued_matching

            return bundled_qernel
        else:
            queued_qernel.append_circuit(new_qernel.circuit)
            queued_qernel.src_qernels.append(new_qernel)

            return 0

class dict2obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [dict2obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, dict2obj(v) if isinstance(v, dict) else v)


class dict2obj_bytes(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k.decode("utf-8"), (list, tuple)):
                setattr(
                    self,
                    k.decode("utf-8"),
                    [
                        dict2obj(x) if isinstance(x, dict) else x
                        for x in v.decode("utf-8")
                    ],
                )
            else:
                setattr(
                    self,
                    k.decode("utf-8"),
                    dict2obj(v.decode("utf-8"))
                    if isinstance(v.decode("utf-8"), dict)
                    else v.decode("utf-8"),
                )


def redisToQPU(qid: int, redisDict: Dict[str, Any]) -> QPU:
    decodedDict = decodeRedisDict(redisDict)
    newqpuInfo = QPU()
    newqpuInfo.id = qid
    newqpuInfo.provider = decodedDict['provider']
    newqpuInfo.name = decodedDict["name"]
    newqpuInfo.alias = decodedDict["alias"]

    try:
        newqpuInfo.local_queue = ast.literal_eval(decodedDict["local_queue"])
    except:
        newqpuInfo.local_queue = []
    
    decodedDict.pop("name")
    newqpuInfo.args = decodedDict
    return newqpuInfo


def redisToInt(redisInt) -> int:
    toReturn = redisInt.decode("utf-8")

    return int(toReturn)


def decodeRedisDict(redisDict: Dict[str, Any]) -> Dict[str, Any]:
    newDict = {}
    for key, value in redisDict.items():
        if key == b"backend":
            newDict[key.decode("utf-8")] = pickle.loads(value)
            continue
        newDict[key.decode("utf-8")] = value.decode("utf-8")
    return newDict


def redisToQernel(jid: int, redisDict: Dict[str, Any]) -> QPU:
    newQernel = Qernel()
    newQernel.id = jid
    # pdb.set_trace()

    try:
        newQernel.status = redisDict[b"status"]
    except:
        newQernel.status = []

    try:
        newQernel.matching = ast.literal_eval(redisDict[b"matching"].decode())
        redisDict.pop(b"matching")
    except:
        newQernel.matching = []
    try:
        newQernel.circuit = redisDict[b"circuit"]
        redisDict.pop(b"circuit")
    except:
        newQernel.circuit = None

    try:
        newQernel.subqernels = ast.literal_eval(redisDict[b"subqernels"].decode())
        redisDict.pop(b"subqernels")
    except:
        newQernel.subqernels = []

    try:
        newQernel.shots = redisDict[b"shots"]
        redisDict.pop(b"shots")
    except:
        newQernel.shots = -1

    redisDict.pop(b"status")
    newQernel.args = redisDict
    return newQernel


def qpuProperties(qpu: int | str | QPU) -> Dict[str, Any]:
    if isinstance(qpu, int):
        tmpqpu = db.getQPU(qpu)
    elif isinstance(qpu, str):
        tmpqpu = db.getQPUFromName(qpu)
    else:
        tmpqpu = qpu

    #print(tmpqpu.alias)
    backend = eval(tmpqpu.name)()
    # backend = db.getQPU_fromname(tmpqpu.alias)
    return backend.properties().to_dict()


def predict_queue_time(qpuId: int) -> int:

    tmpqpu = db.getQPU(qpuId)
    #print(tmpqpu.alias)
    backend = eval(tmpqpu.name)()
    # backend = db.getQPU_fromname(tmpqpu.alias)
    #print(backend.properties().to_dict())

    return 0

def better_estimate_execution_time(qernel: Qernel) -> float:
    #provider = IBMProvider(token=IBM_TOKEN)

    #backends = provider.backends(min_num_qubits=16, filters=lambda b: b.num_qubits <= 127, simulator=False, operational=True)
    
    #backend = provider.get_backend(qernel.match[1])
    #depickle
    backend = db.getQPUField(
        db.getQPUIdFromName(qernel.match[1]),
        "backend")
    backend = pickle.loads(backend)
    #pdb.set_trace()
    
    #backend = [i for i in backends if i.name=='ibm_cairo'][0]

    #backend = fake_provider.FakeKolkataV2()

    #backend = db.getQPUFromName(qernel.match[1])

    tcircuit = transpile(qernel.circuit, backend)

    return CircuitEstimator().estimate_execution_time(tcircuit, backend)

def estimate_execution_time(qernel:Qernel) -> int:

    circ = qernel.circuit.qasm()
    shots = qernel.args["shots"]

    backend = db.getQPUFromName(qernel.match[1])

    avg_gate_times = average_gate_times(qpuProperties(backend))
    
    exec_time = 0
    qc = QuantumCircuit.from_qasm_str(circ)
    qc = backend.transpile(circuit=qc, opt_level=0)

    long_chain_gates = circuit_to_dag(qc).count_ops_longest_path()

    long_chain_gates.pop("measure")
    long_chain_gates.pop("barrier")

    for i, j in long_chain_gates.items():
        exec_time += avg_gate_times[i] * j

    return exec_time*shots


def average_gate_times(hw_properties: Dict):
    gate_times = {}
    gate_counter = {}
    value_pos = 1
    for i in hw_properties["gates"]:

        if i["gate"] == "reset":
            value_pos = 0

        if i["gate"] not in gate_times.keys():
            gate_times[i["gate"]] = i["parameters"][value_pos]["value"]
            gate_counter[i["gate"]] = 1
            continue

        gate_times[i["gate"]] += i["parameters"][value_pos]["value"]
        gate_counter[i["gate"]] += 1

        value_pos = 1

    for i, j in gate_counter.items():
        gate_times[i] = gate_times[i] / j

    return gate_times


def gate_execution_time(hw_properties: Dict, qbits: List[int], gate: str):

    if gate == "id" or gate == "rz" or gate == "sx" or gate == "x" or gate == "reset":
        for i in hw_properties["gates"]:
            if i["qubits"] == qbits and i["gate"] == gate:
                return i["parameters"][1]["value"]

    else:
        for i in hw_properties["gates"]:
            if i["qubits"] == qbits and i["gate"] == gate:
                return i["parameters"][1]["value"]

def load_qpus(qpu_file: str):
    with open(qpu_file + ".yml", "r") as qpuList:
        data = yaml.safe_load(qpuList)

    for i in [j for j in data["qpus"]]:
        newQPU = QPU()
        for x, y in i.items():
            newQPU.args[x] = y

        id = db.addQPU(newQPU)
    return 0