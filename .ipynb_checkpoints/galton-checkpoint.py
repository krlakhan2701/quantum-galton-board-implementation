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
    plt.savefig("figure1.png")
    
from collections import defaultdict

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
plt.savefig("figure2.png")

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
    plt.savefig("figure3.png")

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
plt.savefig("figure4.png")

def create_noise_model():
    return {
        'gate_error_1q': 0.001,    # 1-qubit gate error (0.1%)
        'gate_error_2q': 0.01,     # 2-qubit gate error (1%)
        'decoherence_t1': 50e-6,   # T1 relaxation (50 μs)
        'decoherence_t2': 70e-6,   # T2 dephasing (70 μs)
        'readout_error': 0.02      # Measurement error (2%)
    }

def calculate_fidelity(p, q):
    fidelity = 0.0
    all_keys = set(p.keys()) | set(q.keys())
    for key in all_keys:
        p_val = np.sqrt(p.get(key, 0))
        q_val = np.sqrt(q.get(key, 0))
        fidelity += p_val * q_val
    return fidelity

# Example: After calculating noise model and fidelity
noise_model = {
    'gate_error_1q': 0.001,
    'gate_error_2q': 0.01,
    'decoherence_t1': 50e-6,
    'decoherence_t2': 70e-6,
    'readout_error': 0.02,
}
print("\nNISQ Noise Model Parameters:")
for k, v in noise_model.items():
    print(f"  {k}: {v}")

# Example distributions (replace with your actual output)
p = {'00': 0.5, '01': 0.3, '10': 0.2}
q = {'00': 0.48, '01': 0.35, '10': 0.17}

def calculate_fidelity(p, q):
    import numpy as np
    fidelity = 0.0
    all_keys = set(p.keys()) | set(q.keys())
    for key in all_keys:
        p_val = np.sqrt(p.get(key, 0))
        q_val = np.sqrt(q.get(key, 0))
        fidelity += p_val * q_val
    return fidelity

fi = calculate_fidelity(p, q)
print(f"\nCalculated Fidelity between distributions: {fi:.4f}\n")

import matplotlib.pyplot as plt
import numpy as np

def plot_distributions(p, q):
    labels = list(set(p.keys()) | set(q.keys()))
    p_vals = [p.get(k, 0) for k in labels]
    q_vals = [q.get(k, 0) for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, p_vals, width, label='Distribution P')
    ax.bar(x + width/2, q_vals, width, label='Distribution Q')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Probability')
    ax.set_title('Comparison of Quantum Distributions')
    ax.legend()
    plt.show()


# Example
p = {'00': 0.4, '01': 0.3, '10': 0.3}
q = {'00': 0.38, '01': 0.32, '10': 0.3}
plot_distributions(p, q)
plt.savefig("figure5.png")
