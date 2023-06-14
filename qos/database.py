from typing import Any, Dict, List
import json
import os
from qos.types import Engine, Job, QCircuit
from qos.backends.types import QPU
import redis
from qos.tools import redisToQPU, redisToJob, redisToInt

MAXJOBS = 1000


def qpuIdGen(qid: int):
    return "qpu" + str(qid)


def qcIdGen(qid: int):
    return "qc" + str(qid)


def jobIdGen(qid: int):
    return "job" + str(qid)


def addJob(job: Job) -> int:
    # Add the job to the db
    # This data structure for each job is the follwing:
    # hash - job1 = { <job information>, results: json.dumps(<probabilty distribution>)}
    # hash - job2 = { <job information>, results: json.dumps(<probabilty distribution>)}
    # hash - job3 = { <job information>, results: json.dumps(<probabilty distribution>)}
    # ...
    with redis.Redis() as db:
        newId = db.incr("jobCounter")  # To start from 0
        jobId = jobIdGen(newId)
        for a, b in job.args.items():
            db.hset(jobId, a, b)
    return newId


def setJobField(id: int, key: str, value: float):

    jobId = jobIdGen(id)
    with redis.Redis() as db:
        db.hset(jobId, key, value)
    return 0


def getJobField(id: int, field: str):

    jobId = jobIdGen(id)
    with redis.Redis() as db:
        info = db.hget(
            jobId,
            field,
        )
    return info


def getJob(id: int):

    jobId = jobIdGen(id)
    with redis.Redis() as db:
        all = db.hgetall(jobId)
        job = redisToJob(id, all)
    return job


def addQPU(qpu: QPU) -> int:

    with redis.Redis() as db:
        newId = db.incr("qpuCounter")
        qpuId = qpuIdGen(newId)
        # Add the qpu to the db
        for a, b in qpu.args.items():
            db.hset(qpuId, a, b)

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


def addQC(qc: QCircuit) -> int:

    with redis.Redis() as db:
        newId = db.incr("qcCounter")
        QCId = qcIdGen(newId)
        # Add the qpu to the db
        for a, b in qc.args.items():
            db.hset(QCId, a, b)

    return newId


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


def dumpDB(self):
    pass


def deleteJob(self, qid: int):
    pass


# Not used, maybe for future use
def findFreeQPUId(self):
    with redis.Redis() as db:
        for i in range(MAXJOBS):
            if not db.sismember("qpuList", i):
                return i


# Not used, maybe for future use
def findFreeJobId(self):
    with redis.Redis() as db:
        for i in range(MAXJOBS):
            if not db.sismember("jobList", i):
                return i
