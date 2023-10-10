from qiskit import QuantumCircuit
from qiskit.circuit import Measure, Gate
from qiskit.converters import circuit_to_dag
from qiskit.providers import Backend
from qos.time_estimator.basic_estimator import CircuitEstimator


class GateLengthEstimator(CircuitEstimator):
    """
    Class for estimating job execution time using gate length
    """

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