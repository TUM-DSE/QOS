from typing import Dict, Any, List
import yaml
import pdb
import qos.database as db
from qos.types import Job
from qos.backends.types import QPU
import ast
from qiskit import QuantumCircuit
import logging
from qiskit.converters import circuit_to_dag
from qiskit.providers.fake_provider import *
from qiskit.providers.basicaer import QasmSimulatorPy


def debugPrint():
    subjobs = db.getJobField(1, "subjobs")
    print(subjobs)
    return


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
    newqpuInfo.provider = decodedDict["provider"]
    newqpuInfo.name = decodedDict["name"]
    newqpuInfo.alias = decodedDict["alias"]
    decodedDict.pop("name")
    newqpuInfo.args = decodedDict
    return newqpuInfo


def redisToInt(redisInt) -> int:
    toReturn = redisInt.decode("utf-8")

    return int(toReturn)


def decodeRedisDict(redisDict: Dict[str, Any]) -> Dict[str, Any]:
    newDict = {}
    for key, value in redisDict.items():
        newDict[key.decode("utf-8")] = value.decode("utf-8")
    return newDict


def redisToJob(jid: int, redisDict: Dict[str, Any]) -> QPU:
    newJob = Job()
    newJob.id = jid
    # pdb.set_trace()

    try:
        newJob.status = redisDict[b"status"]
    except:
        newJob.status = []

    try:
        newJob.matching = ast.literal_eval(redisDict[b"matching"].decode())
        redisDict.pop(b"matching")
    except:
        newJob.matching = []
    try:
        newJob.circuit = redisDict[b"circuit"]
        redisDict.pop(b"circuit")
    except:
        newJob.circuit = None

    try:
        newJob.subjobs = ast.literal_eval(redisDict[b"subjobs"].decode())
        redisDict.pop(b"subjobs")
    except:
        newJob.subjobs = []

    try:
        newJob.shots = redisDict[b"shots"]
        redisDict.pop(b"shots")
    except:
        newJob.shots = -1

    redisDict.pop(b"status")
    newJob.args = redisDict
    return newJob


def qpuProperties(qpuId: int):
    tmpqpu = db.getQPU(qpuId)
    print(tmpqpu.alias)
    backend = eval(tmpqpu.name)()
    # backend = db.getQPU_fromname(tmpqpu.alias)
    return backend.properties().to_dict()


def predict_queue_time(qpuId: int) -> int:

    tmpqpu = db.getQPU(qpuId)
    print(tmpqpu.alias)
    backend = eval(tmpqpu.name)()
    # backend = db.getQPU_fromname(tmpqpu.alias)
    print(backend.properties().to_dict())

    return 0


def estimate_execution_time(circ: str, avg_gate_times: Dict, backend: QPU) -> int:
    exec_time = 0
    qc = QuantumCircuit.from_qasm_str(circ)
    qc = backend.transpile(circuit=qc, opt_level=0)

    long_chain_gates = circuit_to_dag(qc).count_ops_longest_path()

    long_chain_gates.pop("measure")
    long_chain_gates.pop("barrier")

    for i, j in long_chain_gates.items():
        exec_time += avg_gate_times[i] * j

    return exec_time


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
