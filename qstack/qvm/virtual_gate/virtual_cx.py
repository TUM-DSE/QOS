from math import pi
from typing import List, Type

from qiskit.circuit.quantumcircuit import QuantumCircuit, Instruction

from vqc.prob import ProbDistribution
from vqc.virtual_gate.virtual_gate import VirtualBinaryGate


class VirtualCX(VirtualBinaryGate):
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

        conf2 = QuantumCircuit(2, 1)
        conf2.rz(pi, 0)
        conf2.h(1)
        conf2.measure(1, 0)
        conf2.h(1)

        conf3 = QuantumCircuit(2, 1)
        conf3.measure(0, 0)
        conf3.h(1)
        conf3.rz(pi, 1)
        conf3.h(1)

        conf4 = QuantumCircuit(2, 1)
        conf4.measure(0, 0)

        conf5 = QuantumCircuit(2, 1)
        conf5.h(1)
        conf5.measure(1, 0)
        conf5.h(1)

        return [conf0, conf1, conf2, conf3, conf4, conf5]

    def knit(self, results: List[ProbDistribution]) -> ProbDistribution:
        r0, _ = results[0].without_first_bit()
        r1, _ = results[1].without_first_bit()
        r20, r21 = results[2].without_first_bit()
        r30, r31 = results[3].without_first_bit()
        r40, r41 = results[4].without_first_bit()
        r50, r51 = results[5].without_first_bit()
        return (r0 + r1 + (r21 - r20) + (r31 - r30) + (r40 - r41) + (r50 - r51)) * 0.5

    def _define(self):
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        self._definition = circuit