import sys

from qiskit_ibm_runtime import QiskitRuntimeService


if len(sys.argv) != 2:
    print("Usage: python scripts/save_ibm_token.py <IBM Quantum API key>")
    exit(1)

TOKEN = sys.argv[1]

QiskitRuntimeService.save_account(channel="ibm_quantum", token=TOKEN, overwrite=True)
