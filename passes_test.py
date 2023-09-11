from qos.distributed_transpiler.analyser import *
from qos.distributed_transpiler.optimiser import *
from qiskit.circuit.random import random_circuit
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import TwoLocal
from qos.types import Qernel
from qvm.qvm.virtual_circuit import generate_instantiations


def test_analyses_passes(qernel: Qernel) -> None:
    basic_pass = BasicAnalysisPass()
    basic_pass.run(qernel)

    supermarq_feature_pass = SupermarqFeaturesAnalysisPass()
    supermarq_feature_pass.run(qernel)

    dg = DependencyGraphFromDAGPass()
    dg.run(qernel)


def test_transformation_passes(qernel: Qernel) -> Qernel:
    bisection_pass = GVBisectionPass(3)

    result = bisection_pass.run(qernel, 10)

    return result

def main():
    #qc = random_circuit(5, 5, max_operands=2, measure=True)
    qc = TwoLocal(5, entanglement='linear', rotation_blocks=["ry"], entanglement_blocks="rzz",reps=1)
    num_params = qc.num_parameters
    qc = qc.bind_parameters(np.random.rand(num_params))
    creg = ClassicalRegister(qc.num_qubits)
    qc.add_register(creg)
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))
    qc = qc.decompose()
    print(qc)

    qernel = Qernel(qc)

    circuits = []
    result = test_transformation_passes(qernel)
    virtual_circuit = VirtualCircuit(result.get_circuit())
    for frag, frag_circuit in virtual_circuit.fragment_circuits.items():
        instance_labels = virtual_circuit.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circuit, instance_labels)
        for c in instantiations:
            print(c)

    #print(qernel.get_metadata())

main()