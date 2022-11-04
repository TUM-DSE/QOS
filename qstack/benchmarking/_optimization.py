import collections
from collections import Counter
from typing import List, Sequence, Tuple, Union

import numpy as np
import scipy
import scipy.optimize as opt
from qiskit.circuit import QuantumCircuit

from ._benchmark import Benchmark
from ._utils import _get_ideal_counts


class VanillaQAOABenchmark(Benchmark):
    """
    Proxy of a full Quantum Approximate Optimization Algorithm (QAOA) benchmark.
    This benchmark targets MaxCut on a Sherrington-Kirkpatrick (SK) model.
    Device performance is given by the Hellinger fidelity between the experimental
    output distribution and the true distribution obtained via scalable, classical
    simulation.
    The ansatz for this QAOA problem follows the typical structure obtained when directly
    translating the objective Hamiltonian to the quantum circuit. Since the SK model is
    completely connected there are O(N^2) interactions that need to take place. These
    are implementation by pairs of CNOTs and Rz rotations between the participating qubits.
    This ansatz is well-suited to QPU architectures which support all-to-all connectivity.
    When a new instance of this benchmark is created, the ansatz parameters will
    be initialized by:
        1. Generating a random instance of an SK graph
        2. Finding approximately optimal angles (rather than random values)
    """

    def __init__(self, num_qubits: int) -> None:
        """Generate a new benchmark instance.
        Args:
            num_qubits: The number of nodes (qubits) within the SK graph.
        """
        self.num_qubits = num_qubits
        self.hamiltonian = self._gen_sk_hamiltonian()
        self.params = self._gen_angles()

    def _get_opt_angles(self) -> Tuple[List, float]:
        def f(params: List) -> float:
            gamma, beta = params
            circ = self._gen_ansatz(gamma, beta)
            probs = _get_ideal_counts(circ)
            objective_value = self._get_expectation_value_from_probs(probs)

            return -objective_value  # because we are minimizing instead of maximizing

        init_params = [np.random.uniform() * 2 * np.pi, np.random.uniform() * 2 * np.pi]
        out = opt.minimize(f, init_params, method="COBYLA")

        return out["x"], out["fun"]

    def _gen_angles(self) -> List:
        # Classically simulate the variational optimization 5 times,
        # return the parameters from the best performing simulation
        best_params, best_cost = [], 10.0
        for _ in range(10):
            params, cost = self._get_opt_angles()
            if cost < best_cost:
                best_params = params
                best_cost = cost
        return best_params

    def _gen_sk_hamiltonian(self) -> List:
        """Randomly pick +1 or -1 for each edge weight."""
        hamiltonian = []
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                hamiltonian.append([i, j, np.random.choice([-1, 1])])

        np.random.shuffle(hamiltonian)

        return hamiltonian

    def _gen_ansatz(self, gamma: float, beta: float) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        # initialize |++++>
        qc.h(qubit=qc.qubits)

        # Apply the phase separator unitary
        for term in self.hamiltonian:
            i, j, weight = term
            phi = gamma * weight

            # Perform a ZZ interaction
            qc.cnot(i, j)
            qc.rz(2 * phi, qubit=j)
            qc.cnot(i, j)

        # Apply the mixing unitary
        qc.rx(2 * beta, qubit=qc.qubits)

        qc.measure_all()

        return qc

    def _get_energy_for_bitstring(self, bitstring: str) -> float:
        energy = 0
        for i, j, weight in self.hamiltonian:
            if bitstring[i] == bitstring[j]:
                energy -= weight  # if edge is UNCUT, weight counts against objective
            else:
                energy += weight  # if edge is CUT, weight counts towards objective
        return energy

    def _get_expectation_value_from_probs(
        self, probabilities: collections.Counter
    ) -> float:
        expectation_value = 0.0
        for bitstring, probability in probabilities.items():
            expectation_value += probability * self._get_energy_for_bitstring(bitstring)
        return expectation_value

    def circuit(self) -> QuantumCircuit:
        """Generate a QAOA circuit for the Sherrington-Kirkpatrick model.
        The ansatz structure is given by the form of the Hamiltonian and requires
        interactions between every pair of qubits. We restrict the depth of this proxy
        benchmark to p=1 to keep the classical simulation scalable.
        """
        gamma, beta = self.params
        return self._gen_ansatz(gamma, beta)

    def score(self, counts: Union[Counter, List[Counter]]) -> float:
        """Compare the experimental output to the output of noiseless simulation.
        The implementation here has exponential runtime and would not scale.
        However, it could in principle be done efficiently via
        https://arxiv.org/abs/1706.02998, so we're good.
        """
        assert isinstance(counts, Counter)

        ideal_counts = _get_ideal_counts(self.circuit())
        total_shots = sum(counts.values())
        experimental_counts = Counter({k: v / total_shots for k, v in counts.items()})

        ideal_value = self._get_expectation_value_from_probs(ideal_counts)
        experimental_value = self._get_expectation_value_from_probs(experimental_counts)

        return 1 - abs(ideal_value - experimental_value) / (2 * ideal_value)


class FermionicSwapQAOABenchmark(Benchmark):
    # TODO
    pass
