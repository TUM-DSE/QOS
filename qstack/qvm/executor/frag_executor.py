import itertools
from typing import Dict, Iterator, Set, Tuple

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Barrier

from vqc.circuit import DistributedCircuit, Fragment
from vqc.device import Device
from vqc.prob import ProbDistribution
from vqc.virtual_gate.virtual_gate import VirtualBinaryGate


class FragmentExecutor:
    _vc: DistributedCircuit
    _fragment: QuantumRegister

    _device: Device

    _results: Dict[Tuple[int, ...], ProbDistribution]
    _not_involved: Set[int]

    def __init__(
        self,
        vc: DistributedCircuit,
        fragment: Fragment,
    ) -> None:
        self._vc = vc
        self._fragment = fragment
        if self._fragment.device is None:
            raise ValueError("fragment has no device")
        self._device = self._fragment.device
        vgates_qubits = [
            set(instr.qubits)
            for instr in vc.data
            if isinstance(instr.operation, VirtualBinaryGate)
        ]
        self._not_involved = set(
            i for i, qubits in enumerate(vgates_qubits) if not (qubits & set(fragment))
        )
        self._results = {}

    def _frag_config_id(self, config_id: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(
            -1 if i in self._not_involved else config_id[i]
            for i in range(len(config_id))
        )

    def _config_ids(self) -> Iterator[Tuple[int, ...]]:
        vgate_instrs = [
            instr
            for instr in self._vc.data
            if isinstance(instr.operation, VirtualBinaryGate)
        ]
        conf_l = [
            tuple(range(len(instr.operation.configure())))
            if set(instr.qubits) & set(self._fragment)
            else (-1,)
            for instr in vgate_instrs
        ]
        return iter(itertools.product(*conf_l))

    def get_result(self, config_id: Tuple[int, ...]) -> ProbDistribution:
        frag_config_id = self._frag_config_id(config_id)
        if frag_config_id not in self._results:
            raise ValueError("no", frag_config_id, config_id)
        return self._results[frag_config_id]

    def execute(self, shots: int = 10000) -> None:
        config_ids = self._config_ids()
        circs = [self._circuit_with_config(config_id) for config_id in config_ids]
        probs = self._device.run(circs, shots)
        for config_id, prob in zip(self._config_ids(), probs):
            self._results[config_id] = prob

    def _circuit_with_config(self, config_id: Tuple[int, ...]) -> QuantumCircuit:
        conf_circ = QuantumCircuit(self._fragment, *self._vc.cregs)
        conf_reg = self._add_config_register(conf_circ, len(config_id))

        ctr = 0
        for instr in self._vc.data:
            if not isinstance(instr.operation, VirtualBinaryGate):
                if isinstance(instr.operation, Barrier):
                    qubits = list(set(instr.qubits) & set(self._fragment))
                    conf_circ.barrier(qubits)
                elif set(instr.qubits) <= set(self._fragment):
                    conf_circ.append(instr.operation, instr.qubits, instr.clbits)
                continue

            if ctr in self._not_involved:
                ctr += 1
                continue
            conf_def = instr.operation.configuration(config_id[ctr])

            if set(instr.qubits) <= set(self._fragment):
                conf_circ.append(
                    conf_def.to_instruction(), instr.qubits, (conf_reg[ctr],)
                )

            elif set(instr.qubits) & set(self._fragment):
                index = 0 if instr.qubits[0] in self._fragment else 1

                conf_circ.append(
                    self._circuit_on_index(conf_def, index).to_instruction(),
                    (instr.qubits[index],),
                    (conf_reg[ctr],),
                )
            else:
                raise RuntimeError("should not happen")

            ctr += 1
        return conf_circ

    @staticmethod
    def _add_config_register(circuit: QuantumCircuit, size: int) -> QuantumRegister:
        num_conf_register = sum(
            1 for creg in circuit.cregs if creg.name.startswith("conf")
        )
        reg = ClassicalRegister(size, name=f"conf_{num_conf_register}")
        circuit.add_register(reg)
        return reg

    @staticmethod
    def _circuit_on_index(circuit: QuantumCircuit, index: int) -> QuantumCircuit:
        qreg = QuantumRegister(1)
        qubit = circuit.qubits[index]
        circ = QuantumCircuit(qreg, *circuit.cregs)
        for instr in circuit.data:
            if len(instr.qubits) == 1 and instr.qubits[0] == qubit:
                circ.append(instr.operation, (qreg[0],), instr.clbits)
        return circ
