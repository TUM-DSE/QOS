from qos.api import QOS
from qos.types import Job
from time import sleep
import pdb
import subprocess


# redis_server = subprocess.Popen(["redis-server"])

# This is an sample client's code


def main():
    # pdb.set_trace()

    qos = QOS()

    newJob = Job()
    Job.args = {"shots": 10}

    newJobId = qos.run(newJob)

    results = qos.results(newJobId)

    while results == 1:
        if results == 1:
            print("Job is still running")

        sleep(0.5)

        results = qos.results(newJobId)

    print(results)


main()

# redis_server.terminate()
