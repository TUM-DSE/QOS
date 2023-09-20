from qos.distributed_transpiler.analyser import *
from qos.distributed_transpiler.optimiser import *

class DistributedTranspiler():
    budget: int
    methods: dict[str, bool]
    size_to_reach: int

    def __init__(self, size_to_reach: int, budget: int = 5, methods: List[str] = []) -> None:
        self.size_to_reach = size_to_reach
        self.budget = budget
        self.methods = {
            "QF": False,
            "GV": False,
            "WC": False,
            "QR": False
        }

        for method in methods:
            assert(self.methods.get(method) is not None)
            self.methods[method] = True

    def estimateOptimalCuttingMethod(self, q: Qernel):
        metadata = q.get_metadata()

        pc = metadata["program_communication"]
        liveness = metadata['liveness']

        if liveness > pc:
            return "WC"
        else:
            return "GV"
        
    def findOptimalCuttingMethod(self, q: Qernel, size_to_reach: int):
        gv_pass = GVOptimalDecompositionPass(size_to_reach)
        wc_pass = OptimalWireCuttingPass(size_to_reach)

        gv_cost = gv_pass.cost(q) ** 6
        wc_cost = wc_pass.cost(q) ** 8

        if gv_cost < wc_cost:
            return "GV"
        else:
            return "WC"
    
    def applyGV(self, q: Qernel, size_to_reach: int):
        if self.methods["GV"]:
            gv_pass = GVOptimalDecompositionPass(size_to_reach)
            
            cost = gv_pass.cost(q)

            if cost <= self.budget:
                q = gv_pass.run(q, self.budget)
                self.budget = self.budget - cost
        
        return q
    
    def applyWC(self, q: Qernel, size_to_reach: int):
        if self.methods["WC"]:
            wc_pass = OptimalWireCuttingPass(size_to_reach)
            
            cost = wc_pass.cost(q)

            if cost <= self.budget:
                q = wc_pass.run(q, self.budget)
                self.budget = self.budget - cost
        
        return q
    
    def applyQR(self, q: Qernel, size_to_reach: int):
        if self.methods["QR"]:
            qr_pass = RandomQubitReusePass(size_to_reach)

            q = qr_pass.run(q)

        return q

    def run(self, q: Qernel):
        #qc = q.get_circuit()
        analysis_pass = BasicAnalysisPass()
        supermarq_features_pass = SupermarqFeaturesAnalysisPass()

        analysis_pass.run(q)
        supermarq_features_pass.run(q)
        #qc_metadata = q.get_metadata()

        if self.methods["QF"]:
            is_qaoa_pass = IsQAOACircuitPass()

            if is_qaoa_pass.run(q):
                qaoa_analysis_pass = QAOAAnalysisPass()
                QF_pass = FrozenQubitsPass(1)

                qaoa_analysis_pass.run(q)
                q = QF_pass.run(q)

                if self.methods["GV"] and self.methods["WC"]:
                    best = self.estimateOptimalCuttingMethod(q)
                    best = self.findOptimalCuttingMethod(q, self.size_to_reach)

                    if best == "GV":
                        q = self.applyGV(q, self.size_to_reach)
                    else:
                        q = self.applyWC(q, self.size_to_reach)   
                
                elif self.methods["GV"]:
                    q = self.applyGV(q, self.size_to_reach)

                elif self.methods["WC"]:
                    q = self.applyWC(q, self.size_to_reach)
        
        elif self.methods["GV"] and self.methods["WC"]:
            best = self.estimateOptimalCuttingMethod(q)
            best = self.findOptimalCuttingMethod(q, self.size_to_reach)

            if best == "GV":
                q = self.applyGV(q, self.size_to_reach)
            else:
                q = self.applyWC(q, self.size_to_reach)   
        
        elif self.methods["GV"]:
             q = self.applyGV(q, self.size_to_reach)
        elif self.methods["WC"]:
            q = self.applyWC(q, self.size_to_reach)
        
        if self.methods["QR"]:
            q = self.applyQR(q, self.size_to_reach)        
        
        return q