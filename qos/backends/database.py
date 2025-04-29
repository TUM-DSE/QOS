from typing import Any, Dict, List
import jsonpickle
import os
from qos.types.types import Engine, Qernel
from qos.backends.types import QPU
from qiskit.providers.fake_provider import *
import logging
import redis
import time
import pdb
import ast
from qos.tools import redisToQPU, redisToQernel, redisToInt, average_gate_times, estimate_execution_time, qpuProperties, better_estimate_execution_time

MAXJOBS = 1000
WINDOWSIZE = 3


def qpuIdGen(qid: int):
    return "qpu" + str(qid)


def qcIdGen(qid: int):
    return "qc" + str(qid)


def qernelIdGen(qid: int):
    return "qernel" + str(qid)


def addQernel(qernel: Qernel) -> int:
    # Add the qernel to the db
    # This data structure for each qernel is the follwing:
    # hash - qernel1 = { <qernel information>, results: json.dumps(<probabilty distribution>)}
    # hash - qernel2 = { <qernel information>, results: json.dumps(<probabilty distribution>)}
    # hash - qernel3 = { <qernel information>, results: json.dumps(<probabilty distribution>)}
    # ...
    with redis.Redis() as db:
        newId = db.incr("qernelCounter")  # To start from 0
        qernelId = qernelIdGen(newId)

        for a, b in qernel.args.items():
            db.hset(qernelId, a, b)

        if qernel.provider != "":
            db.hset(qernelId, "provider", qernel.provider)
        if qernel.circuit != None:
            db.hset(qernelId, "circuit", qernel.circuit.pickle)
        if qernel.matching != "":
            db.hset(qernelId, "matching", str(qernel.matching))
    return newId


def setQernelField(id: int, key: str, value: float):

    qernelId = qernelIdGen(id)
    with redis.Redis() as db:
        db.hset(qernelId, key, value)
    return 0

def addSubqernel(qernelId: int, subqernelId: int):

    with redis.Redis() as db:
        subqernels = db.hget(qernelId, "subqernels")
        if subqernels == None:
            subqernels = [subqernelId]
        else:
            list(subqernels).append(subqernelId)
        
        db.hset(qernelId, "subqernels", str(subqernels))
    return 0


def getQernelField(id: int, field: str):

    qernelId = qernelIdGen(id)
    with redis.Redis() as db:
        info = db.hget(
            qernelId,
            field,
        )
    return info

def getQPUField(id: int|str, field: str):
    
    if type(id) == str:
        id = getQPUIdFromName(id)

    qpuId = qpuIdGen(id)
    with redis.Redis() as db:
        info = db.hget(
            qpuId,
            field,
        )
    return info

def updateQernel(id: int, qernel: Qernel):
    qernelId = qernelIdGen(id)
    with redis.Redis() as db:
        for a, b in qernel.args.items():
            db.hset(qernelId, a, b)
    return 0


def getQernel(id: int):

    qernelId = qernelIdGen(id)
    # pdb.set_trace()
    with redis.Redis() as db:
        all = db.hgetall(qernelId)
        qernel = redisToQernel(id, all)
    return qernel


def addQPU(qpu: QPU) -> int:

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=10)

    with redis.Redis() as db:
        newId = db.incr("qpuCounter")
        qpuId = qpuIdGen(newId)

        db.config_set("notify-keyspace-events", "KEA")

        # Add the qpu to the db
        for a, b in qpu.args.items():
            db.hset(qpuId, a, b)

    #logger.log(10, "Added QPU to the database")

    return newId


def setQPUField(id: int, key: str, value: float):

    qpuId = qpuIdGen(id)
    with redis.Redis() as db:
        db.hset(qpuId, key, value)
    return 0

def reset_local_queues():

    with redis.Redis() as db:
        for i in range(1, redisToInt(db.get("qpuCounter")) + 1):
            #Reset the local queue to None
            db.hdel(qpuIdGen(i), "local_queue")
    return 0


def getLastQPUid():
    with redis.Redis() as db:
        max_qpu_id = db.get("qpuCounter")
        return redisToInt(max_qpu_id)


#def addQC(qc: QCircuit) -> int:
#
#    with redis.Redis() as db:
#        newId = db.incr("qcCounter")
#        QCId = qcIdGen(newId)
#        # Add the qpu to the db
#        for a, b in qc.args.items():
#            db.hset(QCId, a, b)
#
#    return newId


def getQPUFromName(name: str) -> QPU:
    with redis.Redis() as db:
        for i in range(1, redisToInt(db.get("qpuCounter")) + 1):
            qpu = getQPU(i)
            if qpu.alias == name:
                return qpu
    return None


def getQPU(id: int) -> QPU:

    qpuId = qpuIdGen(id)
    with redis.Redis() as db:
        all = db.hgetall(qpuId)
        qpu = redisToQPU(id, all)
    return qpu


def getAllQPU() -> List[QPU]:

    qpus = []
    with redis.Redis() as db:
        for i in range(1, redisToInt(db.get("qpuCounter")) + 1):
            qpus.append(getQPU(i))
    return qpus


def windowCounter():
    with redis.Redis() as db:
        return redisToInt(db.get("windowCounter"))


def dumpDB(self):
    pass


def deleteQernel(self, qid: int):
    pass

def QPU_earliest_free_time(qpu: QPU|str|int):
    #This function returns the earliest time when the qpu will be free

    #last[0] is the execution time, last[1] is the submitted time
    #if last[0]+last[1] > time.time() then the qpu is busy and the ETA is the submitted time of the last qernel plus the execution time, otherwise the ETA is 0
    #If the ETA is more than 0 and the qpu is busy then the submitted time will be the time in the future when the qpu will be free and the incoming qernel is going to be executed
    if type(qpu) == str:
        qpu = getQPUFromName(qpu)
    elif type(qpu) == int:
        qpu = getQPU(qpu)

    if qpu.local_queue == []:
        return 0
    
    last = qpu.local_queue[-1]
    eta = last[1]+last[2]
    return eta


def moveWindow(self):
    with redis.Redis() as db:
        db.decr("windowCounter")
        db.lpop("window")
    return


def addToWindow(self, qernel: Qernel):
    if windowCounter() == WINDOWSIZE:
        moveWindow()

    with redis.Redis() as db:
        db.lpush("window", qernel.id)
    pass


def initWindow(self):
    with redis.Redis() as db:
        db.set("windowCounter", 0)
    return


def currentWindow() -> List[Qernel]:
    with redis.Redis() as db:
        window = db.lrange("window", 0, -1)
    return window


# Not used, maybe for future use
def findFreeQPUId(self):
    with redis.Redis() as db:
        for i in range(MAXJOBS):
            if not db.sismember("qpuList", i):
                return i


# Not used, maybe for future use
def findFreeQernelId(self):
    with redis.Redis() as db:
        for i in range(MAXJOBS):
            if not db.sismember("qernelList", i):
                return i


def getQPUIdFromName(name: str) -> int:
    with redis.Redis() as db:
        for i in range(1, redisToInt(db.get("qpuCounter")) + 1):
            qpu = getQPU(i)
            if qpu.alias == name:
                return i
    return None


def submitQernel(qernel: Qernel) -> int:

    #The match is tuple that contains the choosen qpu, the ideal mapping and the predicted fidelity, by this order
    #The qpu is identified by its name

    qpuId = getQPUIdFromName(qernel.match[1])
    #backend = getQPU(qpuId)
    #pdb.set_trace()

    #The original time is in nanoseconds, it is converted to miliseconds
    #The time() also returns nanoseconds
    #circuit_eta = estimate_execution_time(qernel)/1000000
    #circuit_eta = estimate_execution_time(qernel)
    circuit_eta = better_estimate_execution_time(qernel)
    
    #print("Estimated execution time: " + str(circuit_eta/1000000))

    with redis.Redis() as db:
        local_queue = getQPUField(qpuId, "local_queue")
        if local_queue == None:
            #local_queue = [(qernel.id, circuit_eta, time.time()/1000000)]
            local_queue = [(qernel.id, circuit_eta, qernel.submit_time, 0, qernel.match[2])]
        else:
            local_queue = eval(local_queue)
            #local_queue.append((qernel.id, circuit_eta, time.time()/1000000))
            #If the new qernel submission time is greater than the last qernel submission time plus its ETA then the new qernel execution start time will its submission time, otherwise it will be the last qernel submission time plus its ETA
            if qernel.submit_time > local_queue[-1][2] + local_queue[-1][1]:
                waiting_time = 0
                local_queue.append((qernel.id, circuit_eta, qernel.submit_time, waiting_time, qernel.match[2]))
            else:
                waiting_time = local_queue[-1][2] + local_queue[-1][1] - qernel.submit_time
                local_queue.append((qernel.id, circuit_eta, local_queue[-1][2] + local_queue[-1][1], waiting_time, qernel.match[2]))

        #Info on Local queue 5 values
        # qernel id
        # estimated execution time
        # submission time,
        # estimated waiting time,
        # predicted error

        #print(local_queue)

        setQPUField(qpuId, "local_queue", str(local_queue))

    return circuit_eta  