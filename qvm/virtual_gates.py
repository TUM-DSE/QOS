import abc
from math import cos, pi, sin

from qiskit.circuit import Barrier, Gate, QuantumCircuit, Instruction, QuantumRegister

from qvm.quasi_distr import QuasiDistr


class WireCut(Barrier):
    def __init__(self):
        super().__init__(num_qubits=1, label="wc")

    def _define(self):
        self._definition = QuantumCircuit(1)


class VirtualBinaryGate(Barrier, abc.ABC):
    def __init__(self, original_gate: Gate):
        self._original_gate = original_gate
        super().__init__(num_qubits=original_gate.num_qubits, label=original_gate.name)
        self._name = f"v_{original_gate.name}"
        self._params = original_gate.params
        for inst in self._instantiations():
            self._check_instantiation(inst)

    @property
    def original_gate(self) -> Gate:
        return self._original_gate

    @property
    def num_instantiations(self) -> int:
        return len(self._instantiations())

    @abc.abstractmethod
    def _instantiations(self) -> list[QuantumCircuit]:
        pass

    @abc.abstractmethod
    def knit(self, results: list[QuasiDistr], clbit_idx: int) -> QuasiDistr:
        pass

    def instantiate(self, inst_id: int) -> QuantumCircuit:
        return self._instantiations()[inst_id]

    def _check_instantiation(self, inst: QuantumCircuit):
        assert len(inst.qubits) == 2
        assert len(inst.clbits) == 1
        for instr in inst.data:
            assert len(instr.qubits) == 1
            assert len(instr.clbits) <= 1

    def _define(self):
        circuit = QuantumCircuit(2)
        circuit.append(self._original_gate, [0, 1], [])
        self._definition = circuit


class VirtualMove(VirtualBinaryGate):
    def _instantiations(self) -> list[QuantumCircuit]:
        q0, q1 = 0, 1

        inst0 = QuantumCircuit(2, 1)

        inst1 = QuantumCircuit(2, 1)
        inst1.x(q1)

        inst2 = QuantumCircuit(2, 1)
        inst2.h(q0)
        inst2.measure(q0, 0)
        inst2.h(q1)

        inst3 = QuantumCircuit(2, 1)
        inst3.h(q0)
        inst3.measure(q0, 0)
        inst3.x(q1)
        inst3.h(q1)

        inst4 = QuantumCircuit(2, 1)
        inst4.sdg(q0)
        inst4.h(q0)
        inst4.measure(q0, 0)
        inst4.h(q1)
        inst4.s(q1)

        inst5 = QuantumCircuit(2, 1)
        inst5.sdg(q0)
        inst5.h(q0)
        inst5.measure(q0, 0)
        inst5.x(q1)
        inst5.h(q1)
        inst5.s(q1)

        inst6 = QuantumCircuit(2, 1)
        inst6.measure(q0, 0)

        inst7 = QuantumCircuit(2, 1)
        inst7.measure(q0, 0)
        inst7.x(q1)

        return [inst0, inst1, inst2, inst3, inst4, inst5, inst6, inst7]

    def knit(self, results: list[QuasiDistr], clbit_idx: int) -> QuasiDistr:
        r00, r01 = results[0].split(clbit_idx)
        r10, r11 = results[1].split(clbit_idx)
        r20, r21 = results[2].split(clbit_idx)
        r30, r31 = results[3].split(clbit_idx)
        r40, r41 = results[4].split(clbit_idx)
        r50, r51 = results[5].split(clbit_idx)
        r60, r61 = results[6].split(clbit_idx)
        r70, r71 = results[7].split(clbit_idx)

        return 0.5 * (
            (r00 - r01)
            + (r10 - r11)
            + (r20 - r21)
            - (r30 - r31)
            + (r40 - r41)
            - (r50 - r51)
            + (r60 - r61)
            - (r70 - r71)
        )


class VirtualGateEndpoint(Barrier):
    def __init__(self, virtual_gate: VirtualBinaryGate, vgate_idx: int, qubit_idx: int):
        self._virtual_gate = virtual_gate
        self.vgate_idx = vgate_idx
        self.qubit_idx = qubit_idx
        super().__init__(1, label=f"v_{virtual_gate.name}_{vgate_idx}_{qubit_idx}")

    def instantiate(self, inst_id: int) -> Instruction:
        assert 0 <= inst_id < self._virtual_gate.num_instantiations
        inst = self._virtual_gate.instantiate(inst_id)
        inst_circuit = self._circuit_on_index(inst, self.qubit_idx)
        return inst_circuit.to_instruction()

    @staticmethod
    def _circuit_on_index(circuit: QuantumCircuit, index: int) -> QuantumCircuit:
        qreg = QuantumRegister(1)
        new_circuit = QuantumCircuit(qreg, *circuit.cregs)
        qubit = circuit.qubits[index]
        for instr in circuit.data:
            if len(instr.qubits) == 1 and instr.qubits[0] == qubit:
                new_circuit.append(
                    instr.operation, (new_circuit.qubits[0],), instr.clbits
                )
        return new_circuit


class VirtualCZ(VirtualBinaryGate):
    def _instantiations(self) -> list[QuantumCircuit]:
        inst0 = QuantumCircuit(2, 1)
        inst0.sdg(0)
        inst0.sdg(1)

        inst1 = QuantumCircuit(2, 1)
        inst1.s(0)
        inst1.s(1)

        inst2 = QuantumCircuit(2, 1)
        inst2.measure(0, 0)

        inst3 = QuantumCircuit(2, 1)
        inst3.measure(0, 0)
        inst3.z(1)

        inst4 = QuantumCircuit(2, 1)
        inst4.measure(1, 0)

        inst5 = QuantumCircuit(2, 1)
        inst5.z(0)
        inst5.measure(1, 0)

        return [inst0, inst1, inst2, inst3, inst4, inst5]

    def knit(self, results: list[QuasiDistr], clbit_idx: int) -> QuasiDistr:
        r00, r01 = results[0].split(clbit_idx)
        r10, r11 = results[1].split(clbit_idx)
        r20, r21 = results[2].split(clbit_idx)
        r30, r31 = results[3].split(clbit_idx)
        r40, r41 = results[4].split(clbit_idx)
        r50, r51 = results[5].split(clbit_idx)

        return 0.5 * (
            (r00 - r01)
            + (r10 - r11)
            + (r20 - r21)
            - (r30 - r31)
            + (r40 - r41)
            - (r50 - r51)
        )


class VirtualCX(VirtualCZ):
    def _instantiations(self) -> list[QuantumCircuit]:
        h_gate_circ = QuantumCircuit(2, 1)
        h_gate_circ.h(1)

        cz_insts = []
        for inst in super()._instantiations():
            new_inst = h_gate_circ.compose(inst, inplace=False)
            cz_insts.append(new_inst.compose(h_gate_circ, inplace=False))
        return cz_insts


class VirtualCY(VirtualCX):
    def _instantiations(self) -> list[QuantumCircuit]:
        minus_rz = QuantumCircuit(2, 1)
        minus_rz.rz(-pi / 2, 1)
        plus_rz = QuantumCircuit(2, 1)
        plus_rz.rz(pi / 2, 1)

        cy_insts = []
        for inst in super()._instantiations():
            new_inst = minus_rz.compose(inst, inplace=False)
            cy_insts.append(new_inst.compose(plus_rz, inplace=False))
        return cy_insts


RZZ_ACCURACY = 0.00001


class VirtualRZZ(VirtualBinaryGate):
    def __init__(self, original_gate: Gate):
        super().__init__(original_gate)

    def _instantiations(self) -> list[QuantumCircuit]:
        inst0 = QuantumCircuit(2, 1)

        inst1 = QuantumCircuit(2, 1)
        inst1.z(0)
        inst1.z(1)

        m_theta = -self._params[0]
        if abs(cos(m_theta / 2)) < RZZ_ACCURACY:
            return [inst1]

        if abs(sin(m_theta / 2)) < RZZ_ACCURACY:
            return [inst0]

        inst2 = QuantumCircuit(2, 1)
        inst2.rz(-pi / 2, 0)
        inst2.measure(1, 0)

        inst3 = QuantumCircuit(2, 1)
        inst3.measure(0, 0)
        inst3.rz(-pi / 2, 1)

        inst4 = QuantumCircuit(2, 1)
        inst4.rz(pi / 2, 0)
        inst4.measure(1, 0)

        inst5 = QuantumCircuit(2, 1)
        inst5.measure(0, 0)
        inst5.rz(pi / 2, 1)

        return [inst0, inst1, inst2, inst3, inst4, inst5]

    def knit(self, results: list[QuasiDistr], clbit_idx: int) -> QuasiDistr:
        m_theta = -self._params[0]

        if abs(cos(m_theta / 2)) < RZZ_ACCURACY:
            r, _ = results[0].split(clbit_idx)
            return r * sin(m_theta / 2) ** 2

        if abs(sin(m_theta / 2)) < RZZ_ACCURACY:
            r, _ = results[0].split(clbit_idx)
            return r * cos(m_theta / 2) ** 2

        r0, _ = results[0].split(clbit_idx)
        r1, _ = results[1].split(clbit_idx)

        r23 = results[2] + results[3]
        r45 = results[4] + results[5]

        r230, r231 = r23.split(clbit_idx)
        r450, r451 = r45.split(clbit_idx)

        return (
            (r0 * cos(m_theta / 2) ** 2)
            + (r1 * sin(m_theta / 2) ** 2)
            + (r230 - r231 - r450 + r451) * cos(m_theta / 2) * sin(m_theta / 2)
        )

    def knit_one_state(self, results: list[QuasiDistr], state: str) -> float:
        raise NotImplementedError(
            "knit_one_state is not implemented yet for VirtualRZZ"
        )


class VirtualCPhase(VirtualRZZ):
    def __init__(self, original_gate: Gate):
        super().__init__(original_gate)
        self._params[0] = -self._params[0] / 2

    def _instantiations(self) -> list[QuantumCircuit]:
        lam = self._params[0]
        c1 = QuantumCircuit(2, 1)
        c1.rz(lam / 2, 0)
        c2 = QuantumCircuit(2, 1)
        c2.rz(lam / 2, 1)

        cphase_insts = []
        for inst in super()._instantiations():
            new_inst = c1.compose(inst, inplace=False)
            cphase_insts.append(new_inst.compose(c2, inplace=False))
        return cphase_insts


VIRTUAL_GATE_TYPES: dict[str, type[VirtualBinaryGate]] = {
    "cx": VirtualCX,
    "cy": VirtualCY,
    "cz": VirtualCZ,
    "rzz": VirtualRZZ,
    "cp": VirtualCPhase,
}
