
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit import Aer,IBMQ
from qiskit import execute
from qiskit.tools import visualization
from qiskit.tools.visualization import circuit_drawer, plot_histogram
import matplotlib.pyplot as plt
import warnings
from qiskit.compiler import transpile

### EXECUTE CIRCUIT ###
def execute_circuit(qc, shots=1024, device=None, decimal=False):
    if device is None:
        device = Aer.get_backend('qasm_simulator')
    else:
        device = device
    
    qc_transpiled = transpile(qc, backend=device)
    counts = device.run(qc_transpiled, shots=shots).result().get_counts()
    
    if decimal:
        counts = dict((int(a[::-1],2),b) for (a,b) in counts.items())
    else:
        counts = dict((a[::-1],b) for (a,b) in counts.items())

    return counts

### BASIS STATES PROBABILITIES ###
def basis_states_probs(counts, shots=1024, decimal=False, n_qubits=1):
    probs = []
   
    if decimal:
        basis_states = list(range(2**n_qubits))
    else:
        basis_states = [np.binary_repr(i,width=n_qubits) for i in range(2**n_qubits)]

    for b in basis_states:
        c = counts.get(b)
        if c is None:
            probs.append(0)
        else:
            probs.append(counts[b]/shots)
    
    return probs
