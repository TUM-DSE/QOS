from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import json


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
