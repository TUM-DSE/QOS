from multiprocessing import Process, Value
import time

from qos.error_mitigator.analyser import *
from qos.error_mitigator.optimiser import *

class ErrorMitigator():
    budget: int
    methods: dict[str, bool]
    size_to_reach: int
    ideal_size_to_reach: int

    def __init__(self, size_to_reach: int = 7, ideal_size_to_reach: int = 2, budget: int = 4, methods: List[str] = []) -> None:
        self.size_to_reach = size_to_reach
        self.ideal_size_to_reach = ideal_size_to_reach
        self.budget = budget
        self.methods = {
            "QF": False,
            "GV": False,
            "WC": False,
            "QR": False
        }

        for method in methods:
            self.methods[method] = True

    def estimateOptimalCuttingMethod(self, q: Qernel):
        metadata = q.get_metadata()

        pc = metadata["program_communication"]
        liveness = metadata['liveness']

        if liveness > pc:
            return "WC"
        else:
            return "GV"
        
    def computeCuttingCosts(self, q: Qernel, size_to_reach: int):
        gv_pass = GVOptimalDecompositionPass(size_to_reach)
        wc_pass = OptimalWireCuttingPass(size_to_reach)
        gv_cost = Value("i", 1000)
        wc_cost = Value("i", 1000)

        p = Process(target=gv_pass.cost, args=(q, gv_cost))
        p.start()
        p.join(600)
        if p.is_alive():
            p.terminate()
            p.join()
        gv_cost = gv_cost.value
        #wc_cost = wc_pass.cost(q)
        p = Process(target=wc_pass.cost, args=(q, wc_cost))
        p.start()
        p.join(600)
        if p.is_alive():
            p.terminate()
            p.join()

        wc_cost = wc_cost.value

        return {"GV": gv_cost, "WC": wc_cost}
    
    def applyGV(self, q: Qernel, size_to_reach: int):
        if self.methods["GV"]:
            gv_pass = GVOptimalDecompositionPass(size_to_reach)
            
            #cost = gv_pass.cost(q)

            #if cost <= self.budget:
            q = gv_pass.run(q, self.budget)
        
        return q
    
    def applyWC(self, q: Qernel, size_to_reach: int):
        if self.methods["WC"]:
            wc_pass = OptimalWireCuttingPass(size_to_reach)
            
            #cost = wc_pass.cost(q)

            #if cost <= self.budget:
            q = wc_pass.run(q, self.budget)
        
        return q
    
    def applyQR(self, q: Qernel, size_to_reach: int):
        if self.methods["QR"]:
            qr_pass = RandomQubitReusePass(size_to_reach)

            q = qr_pass.run(q)

        return q

    def run(self, q: Qernel):
        analysis_pass = BasicAnalysisPass()
        supermarq_features_pass = SupermarqFeaturesAnalysisPass()
        
        analysis_pass.run(q)
        supermarq_features_pass.run(q)

        flag = True
        for method in self.methods.values():
            if method:
                flag = False
        
        if flag:
            for k in self.methods.keys():
                self.methods[k] = True

        if self.methods["QF"]:
            is_qaoa_pass = IsQAOACircuitPass()
            budget = self.budget
            if is_qaoa_pass.run(q):
                qaoa_analysis_pass = QAOAAnalysisPass()
                qaoa_analysis_pass.run(q)
                metadata = q.get_metadata()
                num_cnots = metadata["num_nonlocal_gates"]
                hotspots = list(metadata["hotspot_nodes"].values())
                qubits_to_freeze = 0

                for i in range(2):
                    if hotspots[i] / num_cnots >= 0.07:                        
                        qubits_to_freeze = qubits_to_freeze + 1

                qubits_to_freeze = min(qubits_to_freeze, budget)

                if qubits_to_freeze > 0:
                    QF_pass = FrozenQubitsPass(qubits_to_freeze)
                    q = QF_pass.run(q)
                    budget = budget - qubits_to_freeze

            if self.methods["GV"] and self.methods["WC"]:
                size_to_reach = self.size_to_reach
                costs = self.computeCuttingCosts(q, size_to_reach)

                while (costs["GV"] <= budget or costs["WC"] <=budget) and size_to_reach > 2:
                    size_to_reach = size_to_reach - 1
                    costs = self.computeCuttingCosts(q, size_to_reach)
            
                while costs["GV"] > budget and costs["WC"] > budget:
                    size_to_reach = size_to_reach + 1
                    costs = self.computeCuttingCosts(q, size_to_reach)

                if costs["GV"] <= budget or costs["WC"] <= budget:
                    if costs["GV"] <= costs["WC"] or (costs["GV"] == 0 and costs["WC"] == 0):
                        q = self.applyGV(q, size_to_reach)
                    else:
                        q = self.applyWC(q, size_to_reach)          
            elif self.methods["GV"]:
                q = self.applyGV(q, self.size_to_reach)

            elif self.methods["WC"]:
                q = self.applyWC(q, self.size_to_reach)
        
        elif self.methods["GV"] and self.methods["WC"]:
            size_to_reach = self.size_to_reach
            costs = self.computeCuttingCosts(q, size_to_reach)

            while (costs["GV"] <= self.budget or costs["WC"] <= self.budget) and size_to_reach > 2:
                size_to_reach = size_to_reach - 1
                costs = self.computeCuttingCosts(q, size_to_reach)                
            
            while costs["GV"] > self.budget and costs["WC"] > self.budget:
                size_to_reach = size_to_reach + 1
                costs = self.computeCuttingCosts(q, size_to_reach)

            if costs["GV"] <= self.budget or costs["WC"] <= self.budget:
                if costs["GV"] <= costs["WC"] or (costs["GV"] == 0 and costs["WC"] == 0):
                    q = self.applyGV(q, size_to_reach)
                else:
                    q = self.applyWC(q, size_to_reach)
        elif self.methods["GV"]:
             q = self.applyGV(q, self.size_to_reach)
        elif self.methods["WC"]:
            q = self.applyWC(q, self.size_to_reach)
        
        if self.methods["QR"]:
            q = self.applyQR(q, self.size_to_reach)        
        
        return q