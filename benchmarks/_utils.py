from typing import Dict, List, Tuple, Union
from sortedcontainers import SortedDict
from dataclasses import dataclass
from collections import Counter
import numpy as np
import json

from qiskit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity
from qiskit.providers.aer import StatevectorSimulator
from qiskit.quantum_info import Statevector


def _get_ideal_counts(circuit: QuantumCircuit) -> Counter:
    ideal_counts = {}
    sv = Statevector.from_label("0" * circuit.num_qubits)
    circuit_no_meas = circuit.remove_final_measurements(inplace=False)
    sv.evolve(circuit_no_meas)

    for i, amplitude in enumerate(sv):
        bitstring = f"{i:>0{circuit.num_qubits}b}"
        probability = np.abs(amplitude) ** 2
        ideal_counts[bitstring] = probability
    return Counter(ideal_counts)


class ProbDistr(dict):
    def __init__(self, data: Dict[str, float]) -> None:
        super().__init__(data)

    @staticmethod
    def from_counts(counts: Dict[str, int]) -> "ProbDistr":
        shots = sum(counts.values())
        return ProbDistr({k: v / shots for k, v in counts.items()})

    def __add__(self, other: "ProbDistr") -> "ProbDistr":
        pass

    def __sub__(self, other: "ProbDistr") -> "ProbDistr":
        pass


def perfect_counts(original_circuit: QuantumCircuit) -> Dict[str, int]:
    cnt = (
        StatevectorSimulator().run(original_circuit, shots=500000).result().get_counts()
    )
    return {k.replace(" ", ""): v for k, v in cnt.items()}


def fidelity(orginal_circuit: QuantumCircuit, noisy_counts: Dict[str, int]) -> float:
    return hellinger_fidelity(perfect_counts(orginal_circuit), noisy_counts)


class ProbDistribution:
    _probs: SortedDict[int, float]
    _num_meas: int

    def __init__(
        self,
        probs: SortedDict[int, float],
        num_meas: int,
    ) -> None:
        self._probs = probs
        self._num_meas = num_meas

    def __getitem__(self, state: Union[int, str]) -> float:
        if isinstance(state, str):
            base = 2
            if state.startswith("0x"):
                base = 16
            state = int(state, base)
        return self._probs[state]

    def __add__(self, other: "ProbDistribution") -> "ProbDistribution":
        if self._num_meas != other._num_meas:
            raise Exception("The number of measurements must be the same")
        if len(self._probs) == 0:
            return ProbDistribution(other._probs.copy(), self._num_meas)
        if len(other._probs) == 0:
            return ProbDistribution(self._probs.copy(), self._num_meas)

        first = list(self._probs.items())
        second = list(other._probs.items())
        res_probs = SortedDict({})
        i, j = 0, 0
        while i < len(first) and j < len(second):
            if first[i][0] < second[j][0]:
                res_probs[first[i][0]] = first[i][1]
                i += 1
            elif first[i][0] > second[j][0]:
                res_probs[second[j][0]] = second[j][1]
                j += 1
            elif first[i][0] == second[j][0]:
                res_probs[first[i][0]] = first[i][1] + second[j][1]
                i += 1
                j += 1

        while i < len(first):
            res_probs[first[i][0]] = first[i][1]
            i += 1

        while j < len(second):
            res_probs[second[j][0]] = second[j][1]
            j += 1
        return ProbDistribution(res_probs, self._num_meas)

    def __sub__(self, other: "ProbDistribution") -> "ProbDistribution":
        if self._num_meas != other._num_meas:
            raise Exception("The number of measurements must be the same")
        if len(self._probs) == 0:
            new_probs = {state: -p for state, p in other._probs.items()}
            return ProbDistribution(new_probs, self._num_meas)
        if len(other._probs) == 0:
            return ProbDistribution(self._probs.copy(), self._num_meas)

        first = list(self._probs.items())
        second = list(other._probs.items())
        res_probs = SortedDict({})
        i, j = 0, 0
        while i < len(first) and j < len(second):
            if first[i][0] < second[j][0]:
                res_probs[first[i][0]] = first[i][1]
                i += 1
            elif first[i][0] > second[j][0]:
                res_probs[second[j][0]] = -second[j][1]
                j += 1
            elif first[i][0] == second[j][0]:
                res_probs[first[i][0]] = first[i][1] - second[j][1]
                i += 1
                j += 1

        while i < len(first):
            res_probs[first[i][0]] = first[i][1]
            i += 1

        while j < len(second):
            res_probs[second[j][0]] = -second[j][1]
            j += 1
        return ProbDistribution(res_probs, self._num_meas)

    def __mul__(self, other: float) -> "ProbDistribution":
        result = SortedDict({})
        for state, prob in self._probs.items():
            result[state] = prob * other
        return ProbDistribution(result, self._num_meas)

    def counts(self, total_counts: int = 20000) -> Dict[str, int]:
        counts = {}
        for state, prob in self._probs.items():
            counts[bin(state)[2:].zfill(self._num_meas)] = max(
                int(prob * total_counts), 0
            )
        return counts

    @staticmethod
    def from_counts(counts: Dict[str, int]) -> "ProbDistribution":
        base = 2
        some_state = list(counts.keys())[0].replace(" ", "")
        if some_state.startswith("0x"):
            base = 16
        shots = sum(counts.values())
        probs = SortedDict(
            {
                int(state.replace(" ", ""), base): count / shots
                for state, count in counts.items()
            }
        )
        num_meas = len(some_state)
        return ProbDistribution(probs, num_meas)

    def without_first_bit(self) -> Tuple["ProbDistribution", "ProbDistribution"]:
        cmp = 1 << (self._num_meas - 1)
        strip = cmp - 1
        probs1 = SortedDict({})
        probs2 = SortedDict({})
        for state, prob in self._probs.items():
            if state & cmp:
                probs2[state & strip] = prob
            else:
                probs1[state & strip] = prob
        return ProbDistribution(probs1, self._num_meas - 1), ProbDistribution(
            probs2, self._num_meas - 1
        )

    def merge(self, other: "ProbDistribution") -> "ProbDistribution":
        num_meas = max(self._num_meas, other._num_meas)
        res = SortedDict({})
        for state1, prob1 in self._probs.items():
            for state2, prob2 in other._probs.items():
                res[state1 | state2] = prob1 * prob2
        return ProbDistribution(res, num_meas)


@dataclass
class ExecutionStatistic:
    execution_time: float
    merge_time: float
    knit_time: float
    num_executions: int

    def run_time(self) -> float:
        return self.execution_time + self.merge_time + self.knit_time

    def to_json(self) -> Dict[str, float]:
        return {
            "execution_time": self.execution_time,
            "merge_time": self.merge_time,
            "knit_time": self.knit_time,
            "num_executions": float(self.num_executions),
            "run_time": self.run_time(),
        }

    @staticmethod
    def from_json(json_dict: Dict[str, float]) -> "ExecutionStatistic":
        return ExecutionStatistic(
            json_dict["execution_time"],
            json_dict["merge_time"],
            json_dict["knit_time"],
            int(json_dict["num_executions"]),
        )

    def write_to_file(self, file_name: str) -> None:
        with open(file_name, "w") as f:
            json.dump(self.to_json(), f)

    @staticmethod
    def from_file(file_name: str) -> "ExecutionStatistic":
        with open(file_name, "r") as f:
            return ExecutionStatistic.from_json(json.load(f))


def average_runtime(statistics: List[ExecutionStatistic]) -> float:
    return sum(s.run_time() for s in statistics) / len(statistics)


def min_max_runtime(statistics: List[ExecutionStatistic]) -> Tuple[float, float]:
    return min(s.run_time() for s in statistics), max(s.run_time() for s in statistics)
