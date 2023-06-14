from typing import Dict, Any
import yaml
import pdb
import qos.database as db
from qos.types import Job
from qos.backends.types import QPU


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
    newqpuInfo.name = decodedDict["name"]
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
    newJob.status = redisDict["status"]
    redisDict.pop("status")
    newJob.args = redisDict
    return newJob


def load_qpus(qpu_file: str):
    with open(qpu_file + ".yml", "r") as qpuList:
        data = yaml.safe_load(qpuList)

    # Print the data dictionary
    # pdb.set_trace()

    # data = dict2obj(data)

    for i in [j for j in data["qpus"]]:
        newQPU = QPU()
        for x, y in i.items():
            newQPU.args[x] = y

        id = db.addQPU(newQPU)

    return 0
