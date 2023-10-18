import yaml
import qos.database as db
from qos.backends.types import QPU
import logging



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


#print("Available QPUs:")
#for i in range(1, 5):
#    print(db.getQPU(i))

logging.basicConfig(level=50)

load_qpus("qpus_available")
