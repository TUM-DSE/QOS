import sys
import argparse
import pdb
import yaml
import pprint
from qiskit import IBMQ
from benchmarks import *
from backends import IBMQPU
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram, plot_circuit_layout, plot_coupling_map
from collections import Counter
from qiskit.transpiler import Layout
import csv
import requests
import json
import os.path
import time

from qiskit.circuit import QuantumCircuit


class dict2obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [dict2obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, dict2obj(v) if isinstance(v, dict) else v)


def getCNOTS(c: QuantumCircuit) -> int:
    ops = c.count_ops()
    cnots = 0

    for (key, value) in ops.items():
        if key == "cx":
            cnots = value
            break
    return cnots


def layout_from_properties():
    pass


class BackendProperties:
    layout: dict  # Layout of the qubits
    connections: list  # List of gates
    qubits: list  # List of qubits

    def __init__(self) -> None:
        self.qubits = []
        self.connections = []

    def _connections_from_layout(self):
        pass

    def _qubits_from_layout(self):
        pass

    class Qubit:
        t1: float  # Relaxation time, measures how long it takes for the qubit to lose its excited state energy and return to its ground state (ns)
        t2: float  # Dephasing time, measures how long it takes for the quantum phase of the qubit to become randomized. (ns)
        freq: float  # Resonant frequency of the qubit, determines the energy needed to transition between its ground state and excited state (GHz)
        anhar: float  # Anharmonicity: this measures the nonlinearity of the qubit's energy levels, which affects the precision and accuracy of quantum operations.
        # Anharmonicity is typically quantified by the frequency difference between the qubit's ground and first excited states. (GHz)
        rerror: float  # Readout error: this is the probability of incorrectly measuring the state of the qubit during a readout operation. The readout error is
        # typically characterized using a standard set of measurement operations, and it can be affected by various factors such as noise,
        # cross-talk, and interference
        probm0p1: float  # Probability of measuring 0 when prepared in 1
        probm1p0: float  # Probability of measuring 1 when prepared in 0
        rlength: float  # Readout lenght: this is the time duration of the measurement pulse used for reading out the state of the qubit. The readout length is typically
        # optimized to balance the trade-off between measurement accuracy and speed. (ns)
        id_gate: None
        reset_gate: None
        x_gate: None
        sx_gate: None
        rz_gate: None
        connections: list  # List of connections with other qubits, ordered by qubit index

        def __init__(
            self,
            t1,
            t2,
            freq,
            anhar,
            rerror,
            probm0p1,
            probm1p0,
            rlength,
            id_gate,
            reset_gate,
            x_gate,
            sx_gate,
            rz_gate,
            connections,
        ) -> None:
            self.t1 = t1
            self.t2 = t2
            self.freq = freq
            self.anhar = anhar
            self.rerror = rerror
            self.probm0p1 = probm0p1
            self.probm1p0 = probm1p0
            self.rlength = rlength
            self.id_gate = id_gate
            self.reset_gate = reset_gate
            self.x_gate = x_gate
            self.sx_gate = sx_gate
            self.rz_gate = rz_gate
            self.connections = connections

    class Connection:
        q1: Qubit  # Qubit 1
        q2: Qubit  # Qubit 2
        cx: None  # CNOT gate (qubit1 always has smaller index than qubit2)
        cx_inv: None  # Inverse CNOT gate
        jq: float  # Coonection coupling strength (GHz) This is not a gate, its a connection property
        zz: float  # Ising coupling for this connection (GHz) Also, not a gate, its a connection property

        # For these two last properties ususally the lower the better, given that stronger coupling normally
        # increase the noise of the gates and lower the coherence time of the qubits, however some gate types
        # might improve with higher jq or zz coupling, for example the higher the zz value usually improves
        # the entanglement of the qubits, so the Hadamard gate might be better with stronger coupling.

        def __init__(self, qb1, qb2, cx, cx_inv, jq, zz) -> None:
            self.q1 = qb1
            self.q2 = qb2
            self.cx = cx
            self.cx_inv = cx_inv
            self.jq = jq
            self.zz = zz

    class Gate:
        gtype: str  # Gate type
        error: float  # Gate error (unit dependents on the gate type, I think)
        length: float  # Gate length (ns)

        # For these two last properties ususally the lower the better, given that stronger coupling normally
        # increase the noise of the gates and lower the coherence time of the qubits, however some gate types
        # might improve with higher jq or zz coupling, for example the higher the zz value usually improves
        # the entanglement of the qubits, so the Hadamard gate might be better with stronger coupling.

        def __init__(self, gtype, error, length) -> None:
            self.gtype = gtype
            self.error = error
            self.length = length


class Compiler:
    args = []
    backend: IBMQ = None
    provider = None
    props: BackendProperties = None

    def __init__(self) -> None:

        data = self.fetch_properties()

        gates = data["gates"]
        qubits = data["qubits"]
        general = data["general"]

        self.props = BackendProperties()

        pdb.set_trace()

        # Loading qubits
        for i in range(len(qubits)):

            id_gate = BackendProperties.Gate(
                "id",
                gates[i]["parameters"][0]["value"],
                gates[i]["parameters"][1]["value"],
            )

            rz_gate = BackendProperties.Gate(
                "rz",
                gates[len(qubits) + i]["parameters"][0]["value"],
                gates[len(qubits) + i]["parameters"][1]["value"],
            )

            sx_gate = BackendProperties.Gate(
                "sx",
                gates[2 * len(qubits) + i]["parameters"][0]["value"],
                gates[2 * len(qubits) + i]["parameters"][1]["value"],
            )

            x_gate = BackendProperties.Gate(
                "x",
                gates[2 * len(qubits) + i]["parameters"][0]["value"],
                gates[2 * len(qubits) + i]["parameters"][1]["value"],
            )

            reset_gate = BackendProperties.Gate(
                "reset",
                -1,
                gates[-len(qubits) + i]["parameters"][0]["value"],
            )

            connections = []

            for j in range(i + 1, len(qubits)):
                tmp_gate = [
                    gates[w]
                    for w in range(len(gates))
                    if gates[w]["name"] == "cx" + str(i) + "_" + str(j)
                ]
                # out = list(filter(lambda x: x["name"] == "cx0_1", gates[:])) #Alternative filter
                if tmp_gate != []:
                    # If it reaches this point, it should only have one element
                    tmp_gate = tmp_gate[0]
                    cx_gate = BackendProperties.Gate(
                        "cx",
                        tmp_gate["parameters"][0]["value"],
                        tmp_gate["parameters"][1]["value"],
                    )

                    tmp_gate = [
                        gates[w]
                        for w in range(len(gates))
                        if gates[w]["name"] == "cx" + str(j) + "_" + str(i)
                    ][0]
                    cx_inv_gate = BackendProperties.Gate(
                        "cx",
                        tmp_gate["parameters"][0]["value"],
                        tmp_gate["parameters"][1]["value"],
                    )
                    jq = [
                        general[w]
                        for w in range(len(general))
                        if general[w]["name"] == "jq_" + str(i) + str(j)
                        or general[w]["name"] == "jq_" + str(j) + str(i)
                    ][0]
                    zz = [
                        general[w]
                        for w in range(len(general))
                        if general[w]["name"] == "zz_" + str(i) + str(j)
                        or general[w]["name"] == "zz_" + str(j) + str(i)
                    ][0]
                    connections.append(
                        BackendProperties.Connection(
                            i,
                            j,
                            cx_gate,
                            cx_inv_gate,
                            jq["value"],
                            zz["value"],
                        )
                    )

            self.props.qubits.append(
                BackendProperties.Qubit(
                    qubits[i][0]["value"],
                    qubits[i][1]["value"],
                    qubits[i][2]["value"],
                    qubits[i][3]["value"],
                    qubits[i][4]["value"],
                    qubits[i][5]["value"],
                    qubits[i][6]["value"],
                    qubits[i][7]["value"],
                    id_gate,
                    reset_gate,
                    x_gate,
                    sx_gate,
                    rz_gate,
                    connections,
                )
            )

    def fetch_properties(self):
        url = "https://api.quantum-computing.ibm.com/api/Backends/ibmq_belem/properties"

        response = requests.get(url)

        # print(response.json())

        if response.status_code == 200:
            data = response.json()
            with open("properties/ibm_belem_properties.json", "w") as f:
                json.dump(data, f)
            return data
        else:
            print("Error:", response.status_code)
            return None


app = Compiler()
# app.run()
