import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# output as gaussian distribution 

def quantum_galton_board_proper(n_layers: int, shots: int = 2_000):
    n_outputs = n_layers + 1
    n_qubits  = n_outputs + 1          # +1 control/coin qubit
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
    @qml.qnode(dev)
    def circuit():
        # put the “ball” in the centre wire
        centre = n_layers // 2
        qml.PauliX(centre)

        control = n_qubits - 1
        for layer in range(n_layers):
            qml.Hadamard(control)            # coin flip
            for pos in range(n_layers - layer):
                # swap |1> between neighbouring positions if control is |1>
                qml.Toffoli(wires=[control, pos, pos + 1])
            # optional correction CNOTs
            for pos in range(n_layers - layer):
                qml.CNOT(wires=[control, pos])
        return qml.counts(wires=range(n_outputs))

    return circuit()

# quick manual test
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Running Galton board...")
    result = quantum_galton_board_proper(9, shots=5000)
    cleaned = {str(k): int(v) for k, v in result.items()}

    x = [int(bs, 2) for bs in cleaned.keys()]
    y = [v for v in cleaned.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color='skyblue')
    plt.xlabel("Output bin (decimal)")
    plt.ylabel("Counts")
    plt.title("Quantum Galton Board Output Distribution")
    plt.xticks(x)
    plt.grid(axis='y')
    plt.show()
from collections import defaultdict
import matplotlib.pyplot as plt
# result = quantum_galton_board_proper(10, shots=5000)
cleaned = {str(k): int(v) for k, v in result.items()}

# Group by the number of 1s in each bitstring
counts_by_bin = defaultdict(int)
for bitstring, count in cleaned.items():
    bin_index = bitstring.count('1')
    counts_by_bin[bin_index] += count

x = sorted(counts_by_bin.keys())
y = [counts_by_bin[k] for k in x]

plt.figure(figsize=(8,6))
plt.bar(x, y, color='skyblue')
plt.xlabel('Bin Index (number of 1 bits)')
plt.ylabel('Counts')
plt.title('Quantum Galton Board Output Distribution\nGrouped by number of 1 bits')
plt.xticks(x)
plt.grid(axis='y')
plt.show()

#output as exponential distribution
#This circuit creates an approximate exponential-like distribution:

def exponential_distribution(n_qubits: int, lam: float = 1.0, shots: int = 10000):
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def circuit():
        for i in range(n_qubits):
            theta = 2 * np.arcsin(np.exp(-lam * i / n_qubits))
            qml.RY(theta, wires=i)
            if i > 0:
                qml.CRY(2 * np.arcsin(np.exp(-lam * i / n_qubits)), wires=[i-1, i])
        return qml.counts()

    counts = circuit()
    # Convert bitstrings to integers
    x = [int(bs, 2) for bs in counts.keys()]
    y = list(counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color='dodgerblue')
    plt.xlabel("Output (decimal)")
    plt.ylabel("Counts")
    plt.title(f"Exponential Distribution (n_qubits={n_qubits}, λ={lam})")
    plt.grid(axis='y')
    plt.show()

# Example usage
exponential_distribution(n_qubits=8, lam=1.0, shots=10000)


#hadamard_quantum_walk

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def hadamard_quantum_walk(n_steps, shots=2000):
    n_position_qubits = n_steps + 2
    n_total_qubits = n_position_qubits + 1  # +1 for coin
    dev = qml.device('default.qubit', wires=n_total_qubits, shots=shots)

    @qml.qnode(dev)
    def circuit():
        center = n_position_qubits // 2
        qml.PauliX(wires=center)  # Start at center
        coin_qubit = n_total_qubits - 1

        for _ in range(n_steps):
            qml.Hadamard(wires=coin_qubit)
            for pos in range(n_position_qubits - 1):
                qml.CNOT(wires=[coin_qubit, pos+1])
                qml.CNOT(wires=[pos, pos+1])
                qml.CNOT(wires=[coin_qubit, pos+1])
        return qml.counts(wires=list(range(n_position_qubits)))
    return circuit()

# Parameters
n_steps = 20
shots = 10000
result = hadamard_quantum_walk(n_steps, shots)

# Process results
counts_by_position = defaultdict(int)
for bitstring, count in result.items():
    pos_bits = bitstring[:-1]  # Drop coin qubit
    walker_pos = pos_bits[::-1].find('1')  # Find position of '1'
    if walker_pos != -1:
        counts_by_position[walker_pos] += count

# Plot
positions = np.array(sorted(counts_by_position))
frequencies = np.array([counts_by_position[p] for p in positions])

plt.figure(figsize=(10, 6))
plt.bar(positions, frequencies, color='dodgerblue', alpha=0.7, label='Quantum Walk')
plt.xlabel('Walker Position Index')
plt.ylabel('Counts')
plt.title(f'Quantum Hadamard Walk (Steps={n_steps}, Quadratic Spread)')
plt.xticks(positions)
plt.grid(axis='y')

# Theoretical mean/std for quantum walk (optional)
theoretical_std = n_steps / np.sqrt(2)
plt.axvline(n_steps + 1, color='red', linestyle='--', label='Center')
plt.axvline(n_steps + 1 + theoretical_std, color='green', linestyle=':', label='±σ (theory)')
plt.axvline(n_steps + 1 - theoretical_std, color='green', linestyle=':')
plt.legend()
plt.show()