from typing import Any, Dict, List
import json
import os
from qos.types import Engine, Job
import redis


class database:

    # db: redis.Redis
    # idCounter: int

    def __init__(self):
        self.db = redis.Redis(host="localhost", port=6379, db=0)
        self.idCounter = -1
        # I increment it before using it, so the first job will have id 0

    def addJob(job: Job) -> int:
        # Add the job to the db
        # This data structure for each job is the follwing:
        # set - jobList = [jobId1, jobId2, jobId3, ...]
        # hash - jobId1 = { <job information>, results: json.dumps(<probabilty distribution>)}
        # hash - jobId2 = { <job information>, results: json.dumps(<probabilty distribution>)}
        # hash - jobId3 = { <job information>, results: json.dumps(<probabilty distribution>)}
        # ...
        with redis.Redis() as db:
            jobData = {
                "id": 11,
                "status": "submited",
                "results": json.dumps({}),
            }
            db.sadd("jobList", 11)

            for a, b in jobData.items():
                db.hset(11, a, b)

            # self.db.hmset(self.idCounter, jobData)
        return 11

    def setJobField(jobId: int, key: str, value: float):
        with redis.Redis() as db:
            db.hset(jobId, key, value)
        return 0

    def getJobField(jobId: int, field: str):
        with redis.Redis() as db:
            info = db.hget(
                jobId,
                field,
            )
        return info

    def getJob(jobId: int):
        with redis.Redis() as db:
            all = db.hgetall(jobId)
        return all

    def dumpDB(self):
        pass

    def deleteJob(self, qid: int):
        pass

    def findFreeId():
        pass
