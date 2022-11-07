import abc
from collections import Counter
from typing import Callable, List, Optional, Union

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity

from ._benchmark import Benchmark


class ErrorCorrectionBenchmark(Benchmark, abc.ABC):
    def __init__(
        self,
        num_data_qubits: int,
        num_correction_measurement_rounds: int,
        initial_state: Optional[List[int]] = None,
    ):
        self.num_data_qubits = num_data_qubits
        self.num_rounds = num_correction_measurement_rounds
        self.initial_state = (
            initial_state if initial_state is not None else [0] * self.num_data_qubits
        )

        assert (
            len(self.initial_state) == self.num_data_qubits
        ), "Initial state should be the same size as the number of qubits in the circuit."

    def _get_ideal_dist(self) -> Counter:
        """Return the ideal probability distribution of self.circuit().
        Since the only allowed initial states for this benchmark are
        single product states, there is a single bitstring that should be
        measured in the noiseless case.
        """
        ancilla_state, final_state = "", ""
        for i in range(self.num_data_qubits - 1):
            ancilla_state += str(
                (self.initial_state[i] + self.initial_state[i + 1]) % 2
            )
            final_state += str(self.initial_state[i]) + "0"
        else:
            final_state += str(self.initial_state[-1])

        ideal_bitstring = [ancilla_state] * self.num_rounds + [final_state]
        return Counter({"".join(ideal_bitstring): 1.0})

    def score(self, counts: Union[Counter, List[Counter]]) -> float:
        """Device performance is given by the Hellinger fidelity between
        the experimental results and the ideal distribution. The ideal
        is known based on the initial_state parameter.
        """
        assert isinstance(counts, Counter)

        ideal_dist = self._get_ideal_dist()
        total_shots = sum(counts.values())
        experimental_dist = {
            bitstr: shots / total_shots for bitstr, shots in counts.items()
        }
        return hellinger_fidelity(ideal_dist, experimental_dist)


def _error_correction_scaffold_function(
    num_qubits: int,
    correction_measurement_rounds: int,
    initial_state: Union[List[int], str],
    measurement_round_function: Callable[[QuantumCircuit, int], None],
):
    qc = QuantumCircuit(
        2 * num_qubits - 1, (num_qubits - 1) * correction_measurement_rounds
    )

    if isinstance(initial_state, list):
        initial_state = "".join(map(str, initial_state))

    qc.initialize(int(initial_state, 2), [i * 2 for i in range(num_qubits)])
    qc.barrier()

    yield qc

    for round_idx in range(correction_measurement_rounds):
        measurement_round_function(qc, round_idx)
        qc.barrier()

    yield qc

    qc.measure_all()

    yield qc


class BitCodeBenchmark(ErrorCorrectionBenchmark):
    def _measurement_round_bit_code(
        self, quantum_circuit: QuantumCircuit, round_idx: int
    ) -> None:
        num_round_ancilla = self.num_data_qubits - 1

        for c_idx, i in enumerate(range(1, quantum_circuit.num_qubits, 2)):
            quantum_circuit.cx(i - 1, i)
            quantum_circuit.cx(i + 1, i)
            quantum_circuit.measure(i, round_idx * num_round_ancilla + c_idx)
            quantum_circuit.reset(i)

    def circuit(self) -> QuantumCircuit:
        circuit_generator = _error_correction_scaffold_function(
            num_qubits=self.num_data_qubits,
            correction_measurement_rounds=self.num_rounds,
            initial_state=self.initial_state,
            measurement_round_function=self._measurement_round_bit_code,
        )

        # Since the quantum circuit is reference-based,
        # we just return the first yield
        return [circ for circ in circuit_generator][-1]


class PhaseCodeBenchmark(ErrorCorrectionBenchmark):
    def _measurement_round_phase_code(
        self, quantum_circuit: QuantumCircuit, round_idx: int
    ) -> None:
        quantum_circuit.h(quantum_circuit.qubits)
        num_round_ancilla = self.num_data_qubits - 1

        for c_idx, i in enumerate(range(1, quantum_circuit.num_qubits, 2)):
            quantum_circuit.cz(i - 1, i)
            quantum_circuit.cz(i + 1, i)
            quantum_circuit.h(i)
            quantum_circuit.measure(i, round_idx * num_round_ancilla + c_idx)
            quantum_circuit.reset(i)

        quantum_circuit.h([i for i in range(0, quantum_circuit.num_qubits, 2)])

    def circuit(self) -> QuantumCircuit:
        circuit_generator = _error_correction_scaffold_function(
            num_qubits=self.num_data_qubits,
            correction_measurement_rounds=self.num_rounds,
            initial_state=self.initial_state,
            measurement_round_function=self._measurement_round_phase_code,
        )

        qc = next(circuit_generator)

        # Apply Hadamard on data
        qc.h([2 * i for i in range(self.num_data_qubits)])

        qc = next(circuit_generator)

        # Apply Hadamard on data to measure in X-basis
        qc.h([2 * i for i in range(self.num_data_qubits)])

        return next(circuit_generator)
