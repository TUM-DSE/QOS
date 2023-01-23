from typing import Dict, List, Optional, Set

from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit, Barrier
import networkx as nx

from vqc.converters import circuit_to_connectivity_graph
from vqc.device import Device, SimDevice
from vqc.virtual_gate.virtual_gate import VirtualBinaryGate


class Fragment(QuantumRegister):
    device: Device

    def __init__(self, size: int, name: str, device: Optional[Device] = None):
        super().__init__(size, name, None)
        if device is None:
            device = SimDevice()
        self.device = device

    @staticmethod
    def from_qreg(qreg: QuantumRegister, device: Optional[Device] = None) -> "Fragment":
        return Fragment(len(list(qreg)), qreg.name, device)


class DistributedCircuit(QuantumCircuit):
    @staticmethod
    def from_circuit(
        circuit: QuantumCircuit, qubit_groups: Optional[List[Set[Qubit]]] = None
    ) -> "DistributedCircuit":
        if qubit_groups is not None:
            # check qubit-groups
            if set().union(*qubit_groups) != set(circuit.qubits) or bool(
                set().intersection(*qubit_groups)
            ):
                raise ValueError("qubit-groups not valid")

        else:
            con_graph = circuit_to_connectivity_graph(circuit)
            qubit_groups = list(nx.connected_components(con_graph))

        new_frags = [
            Fragment(len(nodes), name=f"frag{i}")
            for i, nodes in enumerate(qubit_groups)
        ]
        qubit_map: Dict[Qubit, Qubit] = {}  # old -> new Qubit
        for nodes, circ in zip(qubit_groups, new_frags):
            node_l = list(nodes)
            for i in range(len(node_l)):
                qubit_map[node_l[i]] = circ[i]

        vc = DistributedCircuit(
            *new_frags,
            *circuit.cregs,
            name=circuit.name,
            global_phase=circuit.global_phase,
            metadata=circuit.metadata,
        )

        for circ_instr in circuit.data:
            vc.append(
                circ_instr.operation,
                [qubit_map[q] for q in circ_instr.qubits],
                circ_instr.clbits,
            )
        return vc

    @property
    def fragments(self) -> List[Fragment]:
        return self.qregs

    @property
    def virtual_gates(self) -> List[VirtualBinaryGate]:
        return [
            instr.operation
            for instr in self.data
            if isinstance(instr.operation, VirtualBinaryGate)
        ]

    def is_valid(self) -> bool:
        for circ_instr in self.data:
            if (
                len(circ_instr.qubits) > 2
                and not isinstance(circ_instr.operation, VirtualBinaryGate)
                and len(set(qubit.register for qubit in circ_instr.qubits)) > 1
            ):
                return False
        return True

    def set_fragment_device(
        self,
        fragment: Fragment,
        device: Device,
    ) -> None:
        reg_index = self.qregs.index(fragment)
        self.qregs[reg_index].device = device

    def fragment_as_circuit(self, fragment: Fragment) -> QuantumCircuit:
        circ = QuantumCircuit(fragment, *self.cregs)
        for instr in self.data:
            if isinstance(instr.operation, Barrier) and set(instr.qubits) & set(
                fragment
            ):
                circ.barrier(list(set(instr.qubits) & set(fragment)))
            if set(instr.qubits) <= set(fragment):
                circ.append(instr.operation, instr.qubits, instr.clbits)
        return circ
