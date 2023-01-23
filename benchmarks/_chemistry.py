import copy
from collections import Counter
from math import cos, pi
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import scipy.optimize as opt
from qiskit.circuit import QuantumCircuit

from ._benchmark import Benchmark
from ._utils import _get_ideal_counts


class HamiltonianSimulationBenchmark(Benchmark):
    def __init__(
        self, num_qubits: int, time_step: int = 1, total_time: int = 1
    ) -> None:
        """Args:
        num_qubits: int
            Size of the TFIM chain, equivalent to the number of qubits.
        time_step: int
            Size of the timestep in attoseconds.
        total_time:
            Total simulation time of the TFIM chain in attoseconds.
        """
        self.num_qubits = num_qubits
        self.time_step = time_step
        self.total_time = total_time

    def circuit(self) -> QuantumCircuit:
        """
        Generate a circuit to simulate the evolution of an n-qubit TFIM
            chain under the Hamiltonian:
            H(t) = - Jz * sum_{i=1}^{n-1}(sigma_{z}^{i} * sigma_{z}^{i+1})
                - e_ph * cos(w_ph * t) * sum_{i=1}^{n}(sigma_{x}^{i})
            where,
                w_ph: frequency of E" phonon in MoSe2.
                e_ph: strength of electron-phonon coupling.
        """
        hbar = 0.658212  # eV*fs
        jz = (
            hbar * pi / 4
        )  # eV, coupling coeff; Jz<0 is antiferromagnetic, Jz>0 is ferromagnetic
        freq = 0.0048  # 1/fs, frequency of MoSe2 phonon

        w_ph = 2 * pi * freq
        e_ph = 3 * pi * hbar / (8 * cos(pi * freq))

        qc = QuantumCircuit(self.num_qubits)

        for step in range(int(self.total_time / self.time_step)):
            t = (step + 0.5) * self.time_step

            # Single qubit terms
            psi = -2.0 * e_ph * cos(w_ph * t) * self.time_step / hbar
            for qubit in qc.qubits:
                qc.h(qubit)
                qc.rz(phi=psi, qubit=qubit)
                qc.h(qubit)

            qc.barrier()

            # Coupling terms
            psi2 = -2.0 * jz * self.time_step / hbar
            for i in range(self.num_qubits - 1):
                qc.cnot(i, i + 1)
                qc.rz(phi=psi2, qubit=i + 1)
                qc.cnot(i, i + 1)

            qc.barrier()

        qc.measure_all()

        return qc

    def _average_magnetization(self, result: dict, shots: int) -> float:
        mag = 0
        for spin_str, count in result.items():
            spin_int = [1 - 2 * int(s) for s in spin_str]
            mag += (
                sum(spin_int) / len(spin_int)
            ) * count  # <Z> weighted by number of times we saw this bitstring
        average_mag = mag / shots  # normalize by the total number of shots
        return average_mag

    def score(self, counts: Union[Counter, Sequence[Counter]]) -> float:
        """Compute the average magnetization of the TFIM chain along the Z-axis
        for the experimental results and via noiseless simulation.
        Args:
            counts: Dictionary of the experimental results. The keys are bitstrings
                represented the measured qubit state, and the values are the number
                of times that state of observed.
        """
        assert isinstance(counts, Counter)

        ideal_counts = _get_ideal_counts(self.circuit())

        total_shots = sum(counts.values())

        mag_ideal = self._average_magnetization(ideal_counts, 1)
        mag_experimental = self._average_magnetization(counts, total_shots)

        return 1 - abs(mag_ideal - mag_experimental) / 2


class VQEBenchmark(Benchmark):
    def __init__(self, num_qubits: int, num_layers: int = 1):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self._params, _ = self._gen_angles()

    def _calc(self, bit_list: List[str], bitstr: str, probs: Counter) -> float:
        energy = 0.0
        for item in bit_list:
            if int(item, base=2).bit_count() == 0:
                energy += probs.get(bitstr, 0)
            else:
                energy -= probs.get(bitstr, 0)
        return energy

    def _get_expectation_value_from_probs(
        self, probs_z: Counter, probs_x: Counter
    ) -> float:
        avg_energy = 0.0

        # Find the contribution to the energy from the X-terms: \sum_i{X_i}
        for bitstr in probs_x.keys():
            bit_list_x = [bitstr[i] for i in range(len(bitstr))]
            avg_energy += self._calc(bit_list_x, bitstr, probs_x)

        # Find the contribution to the energy from the Z-terms: \sum_i{Z_i Z_{i+1}}
        for bitstr in probs_z.keys():
            # fmt: off
            bit_list_z = [bitstr[(i - 1): (i + 1)] for i in range(1, len(bitstr))]
            # fmt: on
            bit_list_z.append(
                bitstr[0] + bitstr[-1]
            )  # Add the wrap-around term manually
            avg_energy += self._calc(bit_list_z, bitstr, probs_z)

        return avg_energy

    def _gen_ansatz(self, params: List[float]) -> List[QuantumCircuit]:
        z_circuit = QuantumCircuit(self.num_qubits)

        param_counter = 0
        for _ in range(self.num_layers):
            # Ry rotation block
            for i in range(self.num_qubits):
                z_circuit.ry(theta=2 * params[param_counter], qubit=i)
                param_counter += 1

            # Rz rotation block
            for i in range(self.num_qubits):
                z_circuit.rz(phi=2 * params[param_counter], qubit=i)
                param_counter += 1

            # Entanglement block
            for i in range(self.num_qubits - 1):
                z_circuit.cnot(i, i + 1)

            # Ry rotation block
            for i in range(self.num_qubits):
                z_circuit.ry(theta=2 * params[param_counter], qubit=i)
                param_counter += 1

            # Rz rotation block
            for i in range(self.num_qubits):
                z_circuit.rz(phi=2 * params[param_counter], qubit=i)
                param_counter += 1

        x_circuit = copy.deepcopy(z_circuit)
        x_circuit.h(x_circuit.qubits)

        # Measure all qubits
        z_circuit.measure_all()
        x_circuit.measure_all()

        return [z_circuit, x_circuit]

    def _gen_angles(self) -> Tuple[List, Any]:
        """Classically simulate the variational optimization and return
        the final parameters.
        """

        def f(params: List) -> float:
            z_circuit, x_circuit = self._gen_ansatz(params)
            z_probs = _get_ideal_counts(z_circuit)
            x_probs = _get_ideal_counts(x_circuit)
            energy = self._get_expectation_value_from_probs(z_probs, x_probs)

            return -energy  # because we are minimizing instead of maximizing

        init_params = [
            np.random.uniform() * 2 * np.pi
            for _ in range(self.num_layers * 4 * self.num_qubits)
        ]
        out = opt.minimize(f, init_params, method="COBYLA")

        return out["x"], out["fun"]

    def circuit(self) -> List[QuantumCircuit]:
        """Construct a parameterized ansatz.
        Returns a list of circuits: the ansatz measured in the Z basis, and the
        ansatz measured in the X basis. The counts obtained from evaluated these
        two circuits should be passed to `score` in the same order they are
        returned here.
        """
        return self._gen_ansatz(self._params)

    def score(self, counts: Union[Counter, List[Counter]]) -> float:
        """Compare the average energy measured by the experiments to the ideal
        value obtained via noiseless simulation. In principle the ideal value
        can be obtained through efficient classical means since the 1D TFIM
        is analytically solvable.
        """
        counts_z, counts_x = counts
        shots_z = sum(counts_z.values())
        probs_z = {bitstr: count / shots_z for bitstr, count in counts_z.items()}
        shots_x = sum(counts_x.values())
        probs_x = {bitstr: count / shots_x for bitstr, count in counts_x.items()}
        experimental_expectation = self._get_expectation_value_from_probs(
            Counter(probs_z),
            Counter(probs_x),
        )

        circuit_z, circuit_x = self.circuit()
        ideal_expectation = self._get_expectation_value_from_probs(
            _get_ideal_counts(circuit_z),
            _get_ideal_counts(circuit_x),
        )

        return float(
            1.0
            - abs(ideal_expectation - experimental_expectation)
            / abs(2 * ideal_expectation)
        )
