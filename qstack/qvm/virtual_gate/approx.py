from math import pi, sin, cos
from typing import List, Type

from qiskit.circuit.quantumcircuit import QuantumCircuit, Instruction
from qiskit.circuit.library.standard_gates import CZGate, CXGate, RZZGate

from vqc.prob import ProbDistribution
from vqc.virtual_gate.virtual_gate import VirtualBinaryGate


class NoneVirtualGate(VirtualBinaryGate):
    def configure(self) -> List[QuantumCircuit]:
        return [QuantumCircuit(2)]

    def knit(self, results: List[ProbDistribution]) -> ProbDistribution:
        return results[0].without_first_bit()[0]


class ApproxVirtualCZ(VirtualBinaryGate):
    def configure(self) -> List[QuantumCircuit]:
        conf0 = QuantumCircuit(2, 1)
        conf0.rz(pi / 2, 0)
        conf0.rz(pi / 2, 1)

        conf1 = QuantumCircuit(2, 1)
        conf1.rz(-pi / 2, 0)
        conf1.rz(-pi / 2, 1)

        return [conf0, conf1]

    def knit(self, results: List[ProbDistribution]) -> ProbDistribution:
        r0, _ = results[0].without_first_bit()
        r1, _ = results[1].without_first_bit()
        return (r0 + r1) * 0.5

    def _define(self):
        circuit = QuantumCircuit(2)
        circuit.cz(0, 1)
        self._definition = circuit


class ApproxVirtualCX(VirtualBinaryGate):
    def configure(self) -> List[QuantumCircuit]:
        conf0 = QuantumCircuit(2, 1)
        conf0.rz(pi / 2, 0)
        conf0.h(1)
        conf0.rz(pi / 2, 1)
        conf0.h(1)

        conf1 = QuantumCircuit(2, 1)
        conf1.rz(-pi / 2, 0)
        conf1.h(1)
        conf1.rz(-pi / 2, 1)
        conf1.h(1)

        return [conf0, conf1]

    def knit(self, results: List[ProbDistribution]) -> ProbDistribution:
        r0, _ = results[0].without_first_bit()
        r1, _ = results[1].without_first_bit()
        return (r0 + r1) * 0.5

    def _define(self):
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        self._definition = circuit


class ApproxVirtualRZZ(VirtualBinaryGate):
    def configure(self) -> List[QuantumCircuit]:
        conf0 = QuantumCircuit(2, 1)

        conf1 = QuantumCircuit(2, 1)
        conf1.z(0)
        conf1.z(1)

        return [conf0, conf1]

    def knit(self, results: List[ProbDistribution]) -> ProbDistribution:
        r0, _ = results[0].without_first_bit()
        r1, _ = results[1].without_first_bit()
        theta = -self.params[0]
        return (r0 * cos(theta / 2) ** 2) + (r1 * sin(theta / 2) ** 2)

    def _define(self):
        circuit = QuantumCircuit(2)
        circuit.rzz(self.params[0], 0, 1)
        self._definition = circuit
