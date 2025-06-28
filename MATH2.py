import numpy as np
import matplotlib.pyplot as plt

# Configuration
data_percent = np.linspace(10, 100, 10)  # % of dataset processed
configs = [
    {"n": 2, "k": 1, "label": "n=2, k=1"},
    {"n": 2, "k": 3, "label": "n=2, k=3"},
    {"n": 4, "k": 3, "label": "n=4, k=3"},
    {"n": 6, "k": 6, "label": "n=6, k=6"},
]

# Constants
T_ser = 0.2                # Fixed serial time
T_par_unit = 0.3           # Time per block at k=1
P_static = 2.0             # Static power in Watts
P_dyn = 0.5                # Dynamic power per core
blocks_total = 100
flops_per_block = 100      # MFLOPs per block (baseline at k=1)

# Compute energy efficiency
def compute_eta(n, k, percent):
    blocks = blocks_total * (percent / 100.0)
    F = flops_per_block * k * blocks
    T = T_ser + (T_par_unit * k * blocks) / min(n, k)
    P = P_static + n * P_dyn
    return F / (T * P)

# Plot
plt.figure(figsize=(10, 6))
for cfg in configs:
    eta_values = [compute_eta(cfg["n"], cfg["k"], p) for p in data_percent]
    plt.plot(data_percent, eta_values, marker='o', label=cfg["label"])

plt.xlabel("Data Processed [%]")
plt.ylabel("Energy Efficiency η(n,k) [MFLOPs/W]")
plt.title("Energy Efficiency vs Data Coverage for MS-AR(k)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
