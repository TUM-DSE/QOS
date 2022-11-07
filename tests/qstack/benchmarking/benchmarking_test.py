from collections import Counter

import pytest

from qstack.benchmarking import (
    BitCodeBenchmark,
    GHZBenchmark,
    HamiltonianSimulationBenchmark,
    PhaseCodeBenchmark,
    VanillaQAOABenchmark,
    VQEBenchmark,
)
from qstack.benchmarking._utils import _get_ideal_counts


# GHZ Tests
def test_ghz_circuit():
    ghz = GHZBenchmark(3)
    assert ghz.circuit().num_qubits == 3


def test_ghz_score():
    ghz = GHZBenchmark(3)
    assert ghz.score(Counter({"000": 500, "111": 500})) == 1

    assert ghz.score(Counter({"000": 250, "111": 500, "011": 250})) != 1


# Error Correction Tests
@pytest.mark.parametrize("class_type", [BitCodeBenchmark, PhaseCodeBenchmark])
def test_code_circuit(class_type):
    instance = class_type(3, 1, [1, 1, 1])
    assert instance.circuit().num_qubits == 5


@pytest.mark.parametrize("class_type", [BitCodeBenchmark, PhaseCodeBenchmark])
def test_code_score(class_type):
    bc = class_type(4, 2, [0, 1, 1, 0])
    assert bc.score(Counter({"1011010010100": 100})) == 1


@pytest.mark.parametrize("class_type", [BitCodeBenchmark, PhaseCodeBenchmark])
def test_invalid_size(class_type):
    with pytest.raises(
        AssertionError,
        match="Initial state should be the same size as the number of qubits in the circuit.",
    ):
        class_type(3, 1, [0])


# Chemistry Tests
def test_hamiltonian_simulation_circuit() -> None:
    hs = HamiltonianSimulationBenchmark(4, 1, 1)
    assert hs.circuit().num_qubits == 4


def test_hamiltonian_simulation_score() -> None:
    hs = HamiltonianSimulationBenchmark(4, 1, 1)
    assert hs._average_magnetization({"1111": 1}, 1) == -1.0
    assert hs._average_magnetization({"0000": 1}, 1) == 1.0
    assert hs.score(_get_ideal_counts(hs.circuit())) > 0.99


def test_vqe_circuit() -> None:
    vqe = VQEBenchmark(3, 1)
    assert len(vqe.circuit()) == 2
    assert vqe.circuit()[0].num_qubits == 3


def test_vqe_score() -> None:
    vqe = VQEBenchmark(3, 1)
    circuits = vqe.circuit()
    probs = [_get_ideal_counts(circ) for circ in circuits]
    assert vqe.score(probs) > 0.99


# Optimization Tests
def test_vanilla_qaoa_circuit():
    qaoa = VanillaQAOABenchmark(4)
    assert qaoa.circuit().num_qubits == 4
    assert qaoa.circuit().count_ops()["cx"] == 12


@pytest.mark.xfail(reason="Issues initializing the parameters of the circuit.")
def test_vanilla_qaoa_score():
    qaoa = VanillaQAOABenchmark(4)
    assert qaoa.score(_get_ideal_counts(qaoa.circuit())) > 0.99
