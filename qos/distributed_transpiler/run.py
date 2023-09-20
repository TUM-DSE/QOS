from qos.distributed_transpiler.analyser import *
from qos.distributed_transpiler.optimiser import *

class DistributedTranspiler():
    budget: int
    methods: {
        "QF": False,
        "GV": False,
        "WC": False,
        "QR": False
    }

    def __init__(self, budget: int = 5, methods: List[str] = []) -> None:
        self.budget = budget

        for method in methods:
            assert(self.methods.get(method) is not None)
            self.methods[method] = True

    def run(self, q: Qernel):
        qc = q.get_circuit()

        if self.methods["QF"]:
            is_qaoa = IsQAOACircuitPass()
            
            if is_qaoa.run(q):
                print("lala")
        
        
        return q