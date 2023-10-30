import yaml
import qos.database as db
from qos.backends.types import QPU

def load_qpus(qpu_file: str):
    with open(qpu_file + ".yml", "r") as qpuList:
        data = yaml.safe_load(qpuList)

    for i in [j for j in data["qpus"]]:
        newQPU = QPU()
        for x, y in i.items():
            newQPU.args[x] = y

        id = db.addQPU(newQPU)

    return 0

load_qpus("evaluation/qpus_available")
