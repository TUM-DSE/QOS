import logging
import multiprocessing
import sys
from enum import Enum
import time
from multiprocessing.pool import ThreadPool
from timeit import default_timer as timer
from typing import Any

import pymoo.gradient.toolbox as anp
from mapomatic import deflate_circuit, matching_layouts
from numpy import argmin
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import StarmapParallelization
from pymoo.core.result import Result
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import (
    BinaryRandomSampling,
    IntegerRandomSampling,
)
from pymoo.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Measure, Reset, Gate
from qiskit.providers import Backend
from qiskit.providers.fake_provider.fake_backend import FakeBackendV2

from src.execution_time.base_estimator import BaseEstimator
from src.execution_time.regression_estimator import RegressionEstimator
from src.optimization.binary_problem import BinarySchedulingProblem
from src.optimization.discrete_problem import DiscreteSchedulingProblem
from src.scheduler.base_scheduler import (
    BaseScheduler,
    Assignment,
    SchedulingJob,
)
from src.utils.benchmark import load_pre_transpiled_circuit

logger = logging.getLogger(__name__)


class TranspilationLevel(Enum):
    """
    Enum for transpilation level
    """

    QPU = "QPU"  # Transpile for each QPU
    PROCESSOR_TYPE = "processor type"  # Transpile for each processor type
    PRE_TRANSPILED = "pre-transpiled"  # Use pre-transpiled circuits


class ProblemType(Enum):
    """
    Enum for problem type
    """

    BINARY = "binary"
    DISCRETE = "discrete"

def calculate_exeucution_time(job, backends, estimator):
        current_execution_times = []

        for i, backend in enumerate(backends):
            if job[i] is None:
                execution_time = sys.maxsize
            else:
                execution_time = estimator.estimate_execution_time(
                    job[i].circuits, backend, shots=job[i].shots
                )
            current_execution_times.append(execution_time)

        return current_execution_times


class MultiObjectiveScheduler(BaseScheduler):
    """
    Circuit scheduler that optimizes for fidelity
    """

    AVERAGE_JOB_TIME = 10

    def __init__(
        self,
        transpilation_count: int = 10,
        transpilation_level: TranspilationLevel = TranspilationLevel.QPU,
        algorithm: Algorithm | None = None,
        estimator: BaseEstimator | None = None,
        problem_type: ProblemType = ProblemType.BINARY,
    ):
        self.transpilation_count = transpilation_count
        self.transpilation_level = transpilation_level
        self.problem_type = problem_type

        self.algorithm = (
            algorithm
            or NSGA2(
                pop_size=100,
                sampling=BinaryRandomSampling(),
                crossover=TwoPointCrossover(),
                mutation=BitflipMutation(),
                eliminate_duplicates=True,
            )
            if problem_type == ProblemType.BINARY
            else NSGA2(
                pop_size=100,
                sampling=IntegerRandomSampling(),
                crossover=SBX(
                    prob=0.5, eta=2.0, vtype=float, repair=RoundingRepair()
                ),
                mutation=PM(
                    prob=0.5, eta=2.0, vtype=float, repair=RoundingRepair()
                ),
                eliminate_duplicates=True,
            )
        )
        self.estimator = estimator or RegressionEstimator()
        self.pre_transpiled_circuits = {}

    def schedule(
        self,
        jobs: list[SchedulingJob],
        backends: list[Backend],
        **kwargs,
    ) -> tuple[list[Assignment], list[SchedulingJob], Any]:
        """
        Schedule a list of jobs to a list of backends
        :param jobs: Jobs to be scheduled
        :param backends: Backends to be scheduled to
        :param kwargs: Additional arguments
        :return: A list of assignments, a list of jobs
        that could not be scheduled and scheduling metadata
        """
        metadata = {}
        # Filter out jobs which include circuits
        # that are too large for the backends
        rejected_jobs = self._filter_jobs(jobs, backends)
        if rejected_jobs and len(rejected_jobs) < len(jobs):
            logger.warning(
                "%d jobs were rejected due to including "
                "circuits that are too large for the backends",
                len(rejected_jobs),
            )
            jobs = [job for job in jobs if job not in rejected_jobs]
        elif rejected_jobs:
            logger.warning(
                "All jobs were rejected due to including "
                "circuits that are too large for the backends"
            )
            return [], jobs, None

        start_transpilation_time = timer()
        start = time.perf_counter()
        transpiled_jobs, fidelities = self._transpile_jobs(jobs, backends)
        total = time.perf_counter() - start
        print(total)

        end_transpilation_time = timer()
        
        metadata["transpilation_time"] = (
            end_transpilation_time - start_transpilation_time
        )
        logger.info(
            "Transpilation took %f seconds", metadata["transpilation_time"]
        )

        # Calculate execution times for each job on each backend
        start = time.perf_counter()
        execution_times = self._calculate_execution_times(
            transpiled_jobs, backends
        )
        total = time.perf_counter() - start
        print(total)

        end_estimation_time = timer()
        
        metadata["estimation_time"] = (
            end_estimation_time - end_transpilation_time
        )
        logger.info(
            "Execution time estimation took %f seconds",
            metadata["estimation_time"],
        )

        backend_sizes = self._get_backend_sizes(backends)
        job_sizes = self._get_job_sizes(jobs)

        # Estimate waiting times for each backend job queue
        start = time.perf_counter()
        waiting_times = self._calculate_backend_queue_waiting_times(backends)
        total = time.perf_counter() - start
        print(total)

        # Define the optimization problem
        start_optimization_time = timer()

        num_threads = int(multiprocessing.cpu_count() / 2)
        pool = ThreadPool(num_threads)
        runner = StarmapParallelization(pool.starmap)
        if self.problem_type == ProblemType.BINARY:
            problem = BinarySchedulingProblem(
                len(jobs),
                len(backends),
                execution_times,
                fidelities,
                waiting_times,
                job_sizes,
                backend_sizes,
                elementwise_runner=runner,
            )
        else:
            problem = DiscreteSchedulingProblem(
                len(jobs),
                len(backends),
                execution_times,
                fidelities,
                waiting_times,
                job_sizes,
                backend_sizes,
                elementwise_runner=runner,
            )

        # Run the optimization
        start = time.perf_counter()
        result = minimize(problem, self.algorithm, verbose=False)
        total = time.perf_counter() - start
        print(total)
        end_optimization_time = timer()
        metadata["optimization_time"] = (
            end_optimization_time - start_optimization_time
        )
        logger.info(
            "Optimization took %f seconds",
            metadata["optimization_time"],
        )

        # Check if a solution was found
        if result.X is None:
            logger.warning("No solution found")
            return [], rejected_jobs + jobs, metadata
        # Get the weights for the objectives
        weights = anp.array(kwargs.get("weights", [0.5, 0.5]))
        # Find the best solution matching the weights
        start_mcdm_time = timer()
        solution_index = self._get_best_solution(result, weights)
        end_mcdm_time = timer()
        metadata["mcdm_time"] = end_mcdm_time - start_mcdm_time
        logger.info(
            "MCDM took %f seconds",
            metadata["mcdm_time"],
        )

        solution = result.X[solution_index]

        # Create the schedule
        start_schedule_generation_time = timer()
        schedule = self._transform_solution_into_schedule(
            solution, transpiled_jobs, backends
        )
        end_schedule_generation_time = timer()
        metadata["schedule_generation_time"] = (
            end_schedule_generation_time - start_schedule_generation_time
        )

        logger.info(
            "Schedule generation took %f seconds",
            metadata["schedule_generation_time"],
        )

        metadata["mean_error"] = result.F[:, 1].tolist()
        metadata["mean_waiting_time"] = result.F[:, 0].tolist()
        metadata["solution_index"] = solution_index
        metadata["solution"] = solution.tolist()

        if self.problem_type == ProblemType.BINARY:
            solution = solution.reshape(len(transpiled_jobs), len(backends))
            solution_execution_times = anp.sum(
                execution_times * solution, axis=1
            )
            all_solutions = result.X.reshape(
                result.X.shape[0], len(transpiled_jobs), len(backends)
            )
            backend_execution_times = anp.sum(
                execution_times * all_solutions, axis=1
            )
            backend_waiting_times = backend_execution_times + waiting_times
            all_solutions_waiting_times = anp.sum(
                backend_waiting_times[:, None, :] * all_solutions, axis=-1
            )
            all_solutions_fidelities = anp.sum(
                fidelities * all_solutions, axis=2
            )
        else:
            assignment_indices = anp.arange(len(jobs))
            solution_execution_times = execution_times[
                assignment_indices, solution
            ]
            backend_execution_times = anp.array(
                [
                    anp.bincount(
                        solution,
                        weights=execution_times[
                            assignment_indices, assignments
                        ],
                        minlength=len(backends),
                    )
                    for assignments in result.X
                ]
            )
            backend_waiting_times = backend_execution_times + waiting_times
            all_solutions_waiting_times = backend_waiting_times[
                anp.arange(result.X.shape[0])[:, None], result.X
            ]
            all_solutions_fidelities = fidelities[assignment_indices, result.X]
        metadata["waiting_time_90_percentile"] = anp.percentile(
            all_solutions_waiting_times, 90, axis=1
        ).tolist()
        metadata["waiting_time_95_percentile"] = anp.percentile(
            all_solutions_waiting_times, 95, axis=1
        ).tolist()
        metadata["fidelity_90_percentile"] = anp.percentile(
            all_solutions_fidelities, 90, axis=1
        ).tolist()
        metadata["fidelity_95_percentile"] = anp.percentile(
            all_solutions_fidelities, 95, axis=1
        ).tolist()

        metadata[
            "solution_execution_times"
        ] = solution_execution_times.tolist()

        return schedule, rejected_jobs, metadata

    def _transform_solution_into_schedule(
        self,
        solution: anp.ndarray,
        transpiled_jobs: list[list[SchedulingJob]],
        backends: list[Backend],
    ) -> list[Assignment]:
        """
        Transform a solution into a schedule
        :param solution: Optimization problem solution
        :param transpiled_jobs: Transpiled jobs
        :param backends: Quantum backends
        :return: A schedule
        """
        schedule = []

        if self.problem_type == ProblemType.BINARY:
            # Reshape the solution to a 2D array of shape
            # (len(jobs), len(backends))
            solution = solution.reshape(len(transpiled_jobs), len(backends))

            for i in range(len(transpiled_jobs)):
                # Find the index of the chosen backend
                backend_index = anp.flatnonzero(solution[i]).item()
                schedule.append(
                    (
                        transpiled_jobs[i][backend_index],
                        backends[backend_index],
                    )
                )
        else:
            for i, backend_index in enumerate(solution):
                schedule.append(
                    (
                        transpiled_jobs[i][backend_index],
                        backends[backend_index],
                    )
                )
        return schedule

    @staticmethod
    def _filter_jobs(
        jobs: list[SchedulingJob], backends: list[Backend]
    ) -> list[SchedulingJob]:
        """
        Filter out jobs that include circuits that are too large for the
        backends
        :param jobs: The jobs to filter
        :param backends: The backends to filter against
        :return: A list of jobs that include circuits that are too large for
        the backends
        """
        max_backend_size = max([backend.num_qubits for backend in backends])
        return [
            job
            for job in jobs
            if any(
                circuit.num_qubits > max_backend_size
                for circuit in job.circuits
            )
        ]

    @staticmethod
    def _normalize_objectives(objectives: anp.ndarray) -> anp.ndarray:
        """
        Normalize the objectives to the range [0, 1]
        :param objectives: The objectives to normalize
        :return: The normalized objectives
        """
        min_objectives = objectives.min(axis=0)
        max_objectives = objectives.max(axis=0)
        return (objectives - min_objectives) / (
            max_objectives - min_objectives
        )

    @staticmethod
    def _get_best_solution(result: Result, weights: anp.ndarray) -> int:
        """
        Find the best solution in the result matching the given weights
        :param result: The result of the optimization
        :param weights: The weights to match
        :return: The index of the best solution
        """
        return int(PseudoWeights(weights).do(result.F))

    def _transpile_circuit(
        self, circuit: QuantumCircuit, backend: Backend
    ) -> tuple[QuantumCircuit | None, float]:
        """
        Transpile a circuit for a given backend
        :param circuit: The circuit to transpile
        :param backend: The backend to transpile for
        :return: The transpiled circuit and the fidelity
        """
        # Check if the circuit can be transpiled for the backend
        if circuit.num_qubits <= backend.num_qubits:
            # Transpile the circuit multiple times due to the stochastic nature
            # of the transpiler
            transpiled_circuits = transpile(
                [circuit] * self.transpilation_count,
                backend=backend,
                optimization_level=3,
            )
            # Choose the circuit with the lowest number of SWAP gates
            swap_gate = set(backend.operation_names).intersection(
                {"cx", "cz", "ecr"}
            )
            if not swap_gate:
                logger.error(
                    "Cannot find swap gate for backend %s",
                    backend.name,
                )
                return None, 0.0
            swap_gate = swap_gate.pop()
            swap_gate_counts = [
                transpiled_circuit.count_ops()[swap_gate]
                for transpiled_circuit in transpiled_circuits
            ]
            best_transpiled_circuit = transpiled_circuits[
                argmin(swap_gate_counts)
            ]

            # Deflate the circuit to remove ancilla qubits
            deflated_circuit = deflate_circuit(best_transpiled_circuit)

            # Find the best layout for the deflated circuit
            layouts = matching_layouts(deflated_circuit, backend.coupling_map)
            if layouts:
                best_layout, fidelity = self._get_best_layout(
                    deflated_circuit, backend, layouts
                )
                best_circuit = transpile(
                    deflated_circuit,
                    backend=backend,
                    initial_layout=best_layout,
                    optimization_level=0,
                )

                return best_circuit, fidelity

        return None, 0.0

    def _get_best_layout(
        self,
        circuit: QuantumCircuit,
        backend: Backend,
        layouts: list[list[int]],
    ) -> tuple[list[int], float]:
        """
        Find the best layout for a circuit on a given backend
        :param circuit: Quantum circuit
        :param backend: Quantum backend
        :param layouts: Possible layouts
        :return: The best layout and its fidelity
        """
        best_layout = None
        best_fidelity = 0.0
        # Calculate the fidelity for each layout and choose the best one
        for layout in layouts:
            fidelity = self._calculate_fidelity(circuit, backend, layout)
            if fidelity > best_fidelity:
                best_layout = layout
                best_fidelity = fidelity

        return best_layout, best_fidelity

    @staticmethod
    def _calculate_fidelity(
        circuit: QuantumCircuit,
        backend: Backend,
        layout: list[int] | None = None,
    ) -> float:
        """
        Calculate the fidelity of a circuit for a given backend and layout
        :param circuit: Quantum circuit
        :param backend: Quantum backend
        :param layout: Layout
        :return: The fidelity
        """
        props = backend.properties()
        fidelity = 1.0
        # Calculate the fidelity for each instruction in the circuit
        for instruction, qargs, cargs in circuit.data:
            # Use the readout error for measurements and resets
            if isinstance(instruction, Measure) or isinstance(
                instruction, Reset
            ):
                qubit = circuit.find_bit(qargs[0]).index
                layout_qubit = layout[qubit] if layout is not None else qubit
                fidelity *= 1 - props.readout_error(layout_qubit)
            # Use the gate error for gates
            elif isinstance(instruction, Gate):
                qubits = [circuit.find_bit(qarg).index for qarg in qargs]
                layout_qubit = (
                    [layout[qubit] for qubit in qubits]
                    if layout is not None
                    else qubits
                )
                fidelity *= 1 - props.gate_error(
                    instruction.name, layout_qubit
                )

        return fidelity

    def _transpile_job(self, job, backends, processor_types):
        # Transpile the job for each backend
        current_transpiled_job = []
        current_fidelities = []
        processor_type_circuits = {}
        for backend in backends:
            transpiled_job = None
            fidelity = 0.0
            if (
                self.transpilation_level
                == TranspilationLevel.PROCESSOR_TYPE
            ):
                # Transpile the circuit for each processor type
                if (
                    processor_types[backend.name]
                    not in processor_type_circuits
                ):
                    transpiled_circuits = []
                    circuit_fidelities = []
                    for circuit in job.circuits:
                        (
                            transpiled_circuit,
                            circuit_fidelity,
                        ) = self._transpile_circuit(circuit, backend)
                        transpiled_circuits.append(transpiled_circuit)
                        circuit_fidelities.append(circuit_fidelity)
                    if all(
                        transpiled_circuit is not None
                        for transpiled_circuit in transpiled_circuits
                    ):
                        transpiled_job = SchedulingJob(
                            transpiled_circuits, job.shots
                        )
                        processor_type_circuits[
                            processor_types[backend.name]
                        ] = transpiled_job
                        fidelity = anp.exp(
                            anp.log(circuit_fidelities).mean()
                        )
                else:
                    transpiled_job = processor_type_circuits[
                        processor_types[backend.name]
                    ]
                    circuit_fidelities = []
                    for transpiled_circuit in transpiled_job.circuits:
                        circuit_fidelity = self._calculate_fidelity(
                            transpiled_circuit, backend
                        )
                        circuit_fidelities.append(circuit_fidelity)
                    fidelity = anp.exp(anp.log(circuit_fidelities).mean())
            elif self.transpilation_level == TranspilationLevel.QPU:
                transpiled_circuits = []
                circuit_fidelities = []
                for circuit in job.circuits:
                    (
                        transpiled_circuit,
                        circuit_fidelity,
                    ) = self._transpile_circuit(circuit, backend)
                    transpiled_circuits.append(transpiled_circuit)
                    circuit_fidelities.append(circuit_fidelity)
                if all(
                    transpiled_circuit is not None
                    for transpiled_circuit in transpiled_circuits
                ):
                    transpiled_job = SchedulingJob(
                        transpiled_circuits, job.shots
                    )
                    fidelity = anp.exp(anp.log(circuit_fidelities).mean())
            elif (
                self.transpilation_level
                == TranspilationLevel.PRE_TRANSPILED
            ):
                transpiled_circuits = [
                    self.pre_transpiled_circuits[
                        (backend.name, circuit.name, circuit.num_qubits)
                    ]
                    for circuit in job.circuits
                ]
                
                if all(
                    transpiled_circuit is not None
                    for transpiled_circuit in transpiled_circuits
                ):
                    transpiled_job = SchedulingJob(
                        transpiled_circuits, job.shots
                    )
                    circuit_fidelities = []
                    for transpiled_circuit in transpiled_job.circuits:
                        circuit_fidelity = self._calculate_fidelity(
                            transpiled_circuit, backend
                        )
                        circuit_fidelities.append(circuit_fidelity)
                    fidelity = anp.exp(anp.log(circuit_fidelities).mean())
            else:
                message = (
                    f"Transpilation level {self.transpilation_level} "
                    f"is not supported"
                )
                logger.error(message)
                raise ValueError(message)
            current_transpiled_job.append(transpiled_job)
            current_fidelities.append(fidelity)

        return (current_transpiled_job, current_fidelities)   


    def _transpile_jobs(
        self,
        jobs: list[SchedulingJob],
        backends: list[Backend],
    ) -> tuple[list[list[SchedulingJob]], anp.ndarray]:
        """
        Transpile a list of jobs for a list of backends
        :param jobs: Jobs to be transpiled
        :param backends: Quantum backends
        :return: The transpiled circuits and the fidelities
        """
        transpiled_jobs = []
        fidelities = []
        values = []
        processor_types = {
            backend.name: backend.processor_type["family"]
            + str(backend.processor_type["revision"])
            + str(backend.processor_type.get("segment", ""))
            for backend in backends
        }

        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 2)) as pool:
            values = pool.starmap(self._transpile_job, [(job, backends, processor_types) for job in jobs])
     
        for v in values:
            transpiled_jobs.append(v[0])
            fidelities.append(v[1])
        """
        for job in jobs:
            # Transpile the job for each backend
            current_transpiled_job = []
            current_fidelities = []
            processor_type_circuits = {}
            for backend in backends:
                transpiled_job = None
                fidelity = 0.0
                if (
                    self.transpilation_level
                    == TranspilationLevel.PROCESSOR_TYPE
                ):
                    # Transpile the circuit for each processor type
                    if (
                        processor_types[backend.name]
                        not in processor_type_circuits
                    ):
                        transpiled_circuits = []
                        circuit_fidelities = []
                        for circuit in job.circuits:
                            (
                                transpiled_circuit,
                                circuit_fidelity,
                            ) = self._transpile_circuit(circuit, backend)
                            transpiled_circuits.append(transpiled_circuit)
                            circuit_fidelities.append(circuit_fidelity)
                        if all(
                            transpiled_circuit is not None
                            for transpiled_circuit in transpiled_circuits
                        ):
                            transpiled_job = SchedulingJob(
                                transpiled_circuits, job.shots
                            )
                            processor_type_circuits[
                                processor_types[backend.name]
                            ] = transpiled_job
                            fidelity = anp.exp(
                                anp.log(circuit_fidelities).mean()
                            )
                    else:
                        transpiled_job = processor_type_circuits[
                            processor_types[backend.name]
                        ]
                        circuit_fidelities = []
                        for transpiled_circuit in transpiled_job.circuits:
                            circuit_fidelity = self._calculate_fidelity(
                                transpiled_circuit, backend
                            )
                            circuit_fidelities.append(circuit_fidelity)
                        fidelity = anp.exp(anp.log(circuit_fidelities).mean())
                elif self.transpilation_level == TranspilationLevel.QPU:
                    transpiled_circuits = []
                    circuit_fidelities = []
                    for circuit in job.circuits:
                        (
                            transpiled_circuit,
                            circuit_fidelity,
                        ) = self._transpile_circuit(circuit, backend)
                        transpiled_circuits.append(transpiled_circuit)
                        circuit_fidelities.append(circuit_fidelity)
                    if all(
                        transpiled_circuit is not None
                        for transpiled_circuit in transpiled_circuits
                    ):
                        transpiled_job = SchedulingJob(
                            transpiled_circuits, job.shots
                        )
                        fidelity = anp.exp(anp.log(circuit_fidelities).mean())
                elif (
                    self.transpilation_level
                    == TranspilationLevel.PRE_TRANSPILED
                ):
                    transpiled_circuits = [
                        self.pre_transpiled_circuits[
                            (backend.name, circuit.name, circuit.num_qubits)
                        ]
                        for circuit in job.circuits
                    ]
                    if all(
                        transpiled_circuit is not None
                        for transpiled_circuit in transpiled_circuits
                    ):
                        transpiled_job = SchedulingJob(
                            transpiled_circuits, job.shots
                        )
                        circuit_fidelities = []
                        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                            circuit_fidelities = pool.starmap(self._calculate_fidelity, [(transpiled_circuit, backend) for transpiled_circuit in transpiled_job.circuits])
                        #for transpiled_circuit in transpiled_job.circuits:
                         #   circuit_fidelity = self._calculate_fidelity(
                          #      transpiled_circuit, backend
                           # )
                            #circuit_fidelities.append(circuit_fidelity)
                        fidelity = anp.exp(anp.log(circuit_fidelities).mean())
                else:
                    message = (
                        f"Transpilation level {self.transpilation_level} "
                        f"is not supported"
                    )
                    logger.error(message)
                    raise ValueError(message)
                current_transpiled_job.append(transpiled_job)
                current_fidelities.append(fidelity)

            transpiled_jobs.append(current_transpiled_job)
            fidelities.append(current_fidelities)

        """


        return transpiled_jobs, anp.array(fidelities)



    def _calculate_execution_times(
        self,
        jobs: list[list[SchedulingJob]],
        backends: [Backend],
    ) -> anp.ndarray:
        """
        Calculate the execution times for a list of jobs on a list of
        backends
        :param jobs: Transpiled jobs
        :param backends: Quantum backends
        :return: The execution times
        """
        execution_times = []

        # Calculate the execution time for each job
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 2)) as pool:
            execution_times = pool.starmap(calculate_exeucution_time, [(job, backends, self.estimator) for job in jobs])
       
        return anp.array(execution_times)

    def _calculate_backend_queue_waiting_times(
        self, backends: [Backend]
    ) -> anp.ndarray:
        """
        Estimate the current job queue waiting times for a list of backends
        :param backends: Quantum backends
        :return: The waiting times
        """
        backend_queue_waiting_times = []

        for backend in backends:
            # If the backend is fake, use the patched method
            if isinstance(backend, FakeBackendV2):
                backend_queue_waiting_times.append(backend.get_waiting_time())
            # Otherwise, use the average job time for estimation
            else:
                backend_queue_waiting_times.append(backend.status().pending_jobs * time)
     
        return anp.array(backend_queue_waiting_times)

    @staticmethod
    def _get_backend_sizes(backends: [Backend]) -> anp.ndarray:
        """
        Get the backend sizes for a list of backends
        :param backends: Quantum backends
        :return: The backend sizes
        """
        return anp.array([backend.num_qubits for backend in backends])

    @staticmethod
    def _get_job_sizes(jobs: list[SchedulingJob]) -> anp.ndarray:
        """
        Get the max job qubit requirements for a list of jobs
        :param jobs: Jobs
        :return: The job sizes
        """
        return anp.array(
            [
                max(circuit.num_qubits for circuit in job.circuits)
                for job in jobs
            ]
        )

    def load_pre_transpiled_circuits(
        self, backends: [Backend], benchmarks: list[str], sizes: list[int]
    ) -> None:
        """
        Load pre-transpiled circuits for a list of backends
        :param backends: Backends to load pre-transpiled circuits for
        :param benchmarks: Benchmarks to load pre-transpiled circuits for
        :param sizes: Sizes to load pre-transpiled circuits for
        """
        self.pre_transpiled_circuits = {}
        self.pre_calculated_fidelities = {}
        for backend in backends:
            for benchmark in benchmarks:
                for size in sizes:
                    self.pre_transpiled_circuits[
                        (backend.name, benchmark, size)
                    ] = load_pre_transpiled_circuit(
                        backend, benchmark_name=benchmark, benchmark_size=size
                    )
