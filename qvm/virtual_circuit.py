import itertools
from multiprocessing.pool import Pool

from qiskit.circuit import (
    ClassicalRegister,
    Barrier,
    QuantumCircuit,
    QuantumRegister as Fragment,
)
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator

from qvm.quasi_distr import QuasiDistr
from qvm.virtual_gates import VirtualBinaryGate, VirtualGateEndpoint, VirtualMove


InstanceLabelType = tuple[int, ...]


class VirtualCircuit:
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._vgate_instrs = [
            instr
            for instr in circuit
            if isinstance(instr.operation, VirtualBinaryGate)
            or isinstance(instr.operation, VirtualMove)
        ]
        # if len(self._vgate_instrs) == 0:
        #     raise ValueError("No virtual gates found in the circuit.")
        self._circuit = self._replace_vgates_with_endpoints(circuit)
        self._frag_circs = {
            qreg: self._circuit_on_fragment(self._circuit, qreg)
            for qreg in circuit.qregs
        }
        self._frag_to_backend = {
            qreg: AerSimulator() for qreg in self._frag_circs.keys()
        }

    def get_instance_labels(self, fragment: Fragment) -> list[InstanceLabelType]:
        if len(self._vgate_instrs) == 0:
            return [()]
        inst_l = [
            tuple(range(len(vg.operation._instantiations())))
            if set(vg.qubits) & set(fragment)
            else (-1,)
            for vg in self._vgate_instrs
        ]
        return list(itertools.product(*inst_l))

    def knit(self, results: dict[Fragment, list[QuasiDistr]], pool: Pool) -> QuasiDistr:
        #if len(self._vgate_instrs) == 0:
            #merged_results = self._fast_merge(results, pool)

        #return merged_results
        merged_results = self._merge(results, pool)

        if len(self._vgate_instrs) == 0:
            return merged_results[0]
        vgates = [instr.operation for instr in self._vgate_instrs]
        clbit_idx = self._circuit.num_clbits + len(vgates) - 1
        while len(vgates) > 0:
            vgate = vgates.pop(-1)
            chunks = _chunk(merged_results, vgate.num_instantiations)
            merged_results = pool.starmap(
                vgate.knit, list(zip(chunks, itertools.repeat(clbit_idx)))
            )
            clbit_idx -= 1
        return merged_results[0]

    @property
    def fragment_circuits(self) -> dict[Fragment, QuantumCircuit]:
        return self._frag_circs.copy()

    def replace_fragment_circuit(
        self, fragment: Fragment, circuit: QuantumCircuit
    ) -> None:
        # TODO checks if fragment and circuit match
        # if set(fragment) != set(circuit.qubits):
        #     raise ValueError("Fragment and circuit do not match.")
        self._frag_circs[fragment] = circuit
        
    def get_backend(self, fragment: Fragment) -> BackendV2:
        if fragment not in self._frag_to_backend:
            raise ValueError("Fragment not found.")
        return self._frag_to_backend[fragment]

    def set_backend(self, fragment: Fragment, backend: BackendV2) -> None:
        if fragment not in self._frag_to_backend:
            raise ValueError("Fragment not found.")
        self._frag_to_backend[fragment] = backend

    @staticmethod
    def _replace_vgates_with_endpoints(circuit: QuantumCircuit) -> QuantumCircuit:
        new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        vgate_index = 0
        for instr in circuit:
            op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
            if isinstance(op, VirtualBinaryGate):
                for i in range(2):
                    new_circuit.append(
                        VirtualGateEndpoint(op, vgate_idx=vgate_index, qubit_idx=i),
                        [qubits[i]],
                        [],
                    )
                vgate_index += 1
                continue
            new_circuit.append(op, qubits, clbits)
        return new_circuit

    @staticmethod
    def _circuit_on_fragment(
        circuit: QuantumCircuit, fragment: Fragment
    ) -> QuantumCircuit:
        new_circuit = QuantumCircuit(fragment, *circuit.cregs)
        for instr in circuit.data:
            op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
            if set(qubits) <= set(fragment):
                new_circuit.append(op, qubits, clbits)
                continue
            elif isinstance(op, Barrier):
                continue
            elif set(qubits) & set(fragment):
                raise ValueError(
                    f"Circuit contains gates that act on multiple fragments. {op}"
                )
        return new_circuit

    def _global_inst_labels(self) -> list[InstanceLabelType]:
        inst_l = [
            range(len(vg.operation._instantiations())) for vg in self._vgate_instrs
        ]
        return list(itertools.product(*inst_l))

    def _global_to_fragment_inst_label(
        self, fragment: Fragment, global_inst_label: tuple[int, ...]
    ) -> tuple[int, ...]:
        frag_inst_label = []
        for i, vg_instr in enumerate(self._vgate_instrs):
            if set(vg_instr.qubits) & set(fragment):
                frag_inst_label.append(global_inst_label[i])
            else:
                frag_inst_label.append(-1)
        return tuple(frag_inst_label)

    def _fragment_results(
        self, fragment: Fragment, results: list[QuasiDistr]
    ) -> list[QuasiDistr]:
        labeled_results = dict(zip(self.get_instance_labels(fragment), results))

        frag_results = []
        for global_label in self._global_inst_labels():
            frag_instance_label = self._global_to_fragment_inst_label(
                fragment, global_label
            )

            frag_results.append(labeled_results[frag_instance_label])

        return frag_results

    def _merge(
        self, results: dict[Fragment, list[QuasiDistr]], pool: Pool
    ) -> list[QuasiDistr]:
        distr_lists = [
            self._fragment_results(frag, distrs) for frag, distrs in results.items()
        ]
        return _merge_distr_lists(tuple(distr_lists), pool)
    
    def _fast_merge(self, results: dict[Fragment, list[QuasiDistr]], pool: Pool) -> list[QuasiDistr]:
        for frag, distr in results.items():
            print(distr)
        exit()
        distr_lists = [
            self._fragment_results(frag, distrs) for frag, distrs in results.items()
        ]



def generate_instantiations(
    fragment_circuit: QuantumCircuit,
    inst_labels: list[InstanceLabelType],
) -> list[QuantumCircuit]:
    return [
        _instantiate_fragment(fragment_circuit, inst_label)
        for inst_label in inst_labels
    ]


def _chunk(lst: list, n: int) -> list[list]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _instantiate_fragment(
    fragment_circuit: QuantumCircuit, inst_label: InstanceLabelType
) -> QuantumCircuit:
    if len(inst_label) == 0:
        return fragment_circuit.copy()
    config_register = ClassicalRegister(len(inst_label), "vgate_c")
    new_circuit = QuantumCircuit(
        *fragment_circuit.qregs, *(fragment_circuit.cregs + [config_register])
    )
    for instr in fragment_circuit:        
        op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
        if isinstance(op, VirtualGateEndpoint):
            vgate_idx = op.vgate_idx
            op = op.instantiate(inst_label[vgate_idx])
            clbits = [config_register[vgate_idx]]
        new_circuit.append(op, qubits, clbits)
    return new_circuit.decompose()


def _merge_distrs(distrs: tuple[QuasiDistr, ...]) -> QuasiDistr:
    assert len(distrs) > 0
    merged = distrs[0]
    for res in distrs[1:]:  # type: ignore
        merged = merged.merge(res)
    return merged


def _merge_distr_lists(
    distr_lists: tuple[list[QuasiDistr], ...], pool: Pool
) -> list[QuasiDistr]:
    workload = zip(*distr_lists)
    return pool.map(_merge_distrs, workload)
