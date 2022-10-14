from math import pi, sin, cos
from typing import List, Type

from qiskit.circuit.quantumcircuit import QuantumCircuit, Instruction

from vqc.prob import ProbDistribution
from vqc.virtual_gate.virtual_gate import VirtualBinaryGate


class VirtualRZZ(VirtualBinaryGate):
    def configure(self) -> List[QuantumCircuit]:
        conf0 = QuantumCircuit(2, 1)

        conf1 = QuantumCircuit(2, 1)
        conf1.z(0)
        conf1.z(1)

        conf2 = QuantumCircuit(2, 1)
        conf2.rz(-pi / 2, 0)
        conf2.measure(1, 0)

        conf3 = QuantumCircuit(2, 1)
        conf3.measure(0, 0)
        conf3.rz(-pi / 2, 1)

        conf4 = QuantumCircuit(2, 1)
        conf4.rz(pi / 2, 0)
        conf4.measure(1, 0)

        conf5 = QuantumCircuit(2, 1)
        conf5.measure(0, 0)
        conf5.rz(pi / 2, 1)

        return [conf0, conf1, conf2, conf3, conf4, conf5]

    def knit(self, results: List[ProbDistribution]) -> ProbDistribution:
        r0, _ = results[0].without_first_bit()
        r1, _ = results[1].without_first_bit()
        r23 = results[2] + results[3]
        r45 = results[4] + results[5]

        r230, r231 = r23.without_first_bit()
        r450, r451 = r45.without_first_bit()

        theta = -self.params[0]
        return (
            (r0 * cos(theta / 2) ** 2)
            + (r1 * sin(theta / 2) ** 2)
            + (r230 - r231 - r450 + r451) * cos(theta / 2) * sin(theta / 2)
        )

    def _define(self):
        circuit = QuantumCircuit(2)
        circuit.rzz(self.params[0], 0, 1)
        self._definition = circuit
