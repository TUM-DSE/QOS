from typing import Dict


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
