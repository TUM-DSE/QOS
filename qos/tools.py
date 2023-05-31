from typing import Dict, Any
import yaml
import pdb
import qos.database
from qos.backends.types import QPUInfo


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


def redisToObj(redisDict: Dict[str, Any]):
    job = dict2obj_bytes(redisDict)
    return job


def load_qpus(qpu_file: str):
    with open(qpu_file + ".yml", "r") as qpuList:
        data = yaml.safe_load(qpuList)

    # Print the data dictionary
    print(data)
    print("\n")

    pdb.set_trace()

    # data = dict2obj(data)

    for i in [j for j in data["qpus"]]:
        newQPU = QPUInfo()
        for x, y in i.items():
            newQPU.args[x] = y

        qos.database.addQPU(newQPU)

    return 0
