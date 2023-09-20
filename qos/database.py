from typing import Any, Dict, List
import jsonpickle
import os
from qos.types import Engine, Qernel
from qos.backends.types import QPU
import logging
import redis
import pdb
from qos.tools import redisToQPU, redisToQernel, redisToInt

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
            db.hset(qernelId, "circuit", qernel.circuit.qasm())
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

    logger.log(10, "Added QPU to the database")

    return newId


def setQPUField(id: int, key: str, value: float):

    qpuId = qpuIdGen(id)
    with redis.Redis() as db:
        db.hset(qpuId, key, value)
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


def getQPU_fromname(name: str) -> QPU:
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
