from abc import abstractmethod
from typing import TypeAlias

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit import QuantumCircuit
from qiskit.circuit import Measure, Gate
from qiskit.converters import circuit_to_dag
from qiskit.providers import Backend

from qos.time_estimator.base_estimator import BaseEstimator

Assignment: TypeAlias = tuple[QuantumCircuit, Backend]


class CircuitEstimator(BaseEstimator):
    """
    Base class for estimating job execution time using individual circuit
    execution time estimations
    """

    def estimate_execution_time(
        self,
        circuits: list[QuantumCircuit],
        backend: Backend,
        **kwargs,
    ) -> float:
        """
        Estimate the execution time of a quantum job on a specified backend
        :param circuits: Circuits in the quantum job
        :param backend: Backend to be executed on
        :param kwargs: Additional arguments, like the run configuration
        :return: Estimated execution time
        """
        execution_time = 0
        rep_delay = kwargs.get("rep_delay", backend.default_rep_delay)
        shots = min(kwargs.get("shots", 8192), backend.max_shots)
        #for circuit in circuits:
        execution_time += (self.estimate_circuit_execution_time(circuits, backend)+rep_delay)
        execution_time *= shots

        return execution_time

    def estimate_circuit_execution_time(
        self, circuit: QuantumCircuit, backend: Backend
    ) -> float:
        """
        Estimate the execution time of a single circuit on a backend using gate
        and measurement lengths
        :param circuit: Circuit to be estimated
        :param backend: Backend to be estimated on
        :return: Estimated execution time
        """
        backend_properties = backend.properties()
        execution_time = 0
        dag = circuit_to_dag(circuit, copy_operations=False)
        for layer in dag.layers():
            max_operation_time = 0
            for node in layer["graph"].op_nodes():
                operation_time = 0
                if isinstance(node.op, Measure):
                    qubit = circuit.find_bit(node.qargs[0]).index
                    operation_time = backend_properties.readout_length(qubit)
                elif isinstance(node.op, Gate):
                    qubits = [
                        circuit.find_bit(qarg).index for qarg in node.qargs
                    ]
                    operation_time = backend_properties.gate_length(
                        node.op.name, qubits
                    )
                max_operation_time = max(max_operation_time, operation_time)
            execution_time += max_operation_time
        return execution_time
        """
        Estimate the execution time of a single circuit on a backend
        :param circuit: Circuit to be estimated
        :param backend: Backend to be estimated on
        :return: Estimated execution time
        """
        ...