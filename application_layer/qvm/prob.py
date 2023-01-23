from typing import Dict, Tuple, Union

from sortedcontainers import SortedDict


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
