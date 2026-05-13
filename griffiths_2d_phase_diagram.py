import os
import random
from datetime import datetime
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from griffiths_simulator import SISGriffithsSimulator


def get_steady_state(matrix_adj, lambda_max, mu, steps, num_seeds, seed, N):
    """Run simulation for a given lambda_max and return the steady-state prevalence."""
    steady_rho_runs = []

    for s in range(num_seeds):
        current_seed = seed + s
        np.random.seed(current_seed)
        random.seed(current_seed)

        betas = np.random.uniform(0, lambda_max, size=N)
        local_sim = SISGriffithsSimulator()

        # We start fully infected and average the last 20% of the steps
        rho_history = local_sim.run(matrix_adj, betas, mu, steps, initial_fraction=1.0)
        steady_rho = np.mean(rho_history[int(steps * 0.8):])
        steady_rho_runs.append(steady_rho)

    return np.mean(steady_rho_runs)


print("=" * 60)
print("       SIS Model - 2D Phase Diagram Generation")
print("=" * 60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("results_griffiths", f"phase_diagram_2d_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
N = 20000  # Moderately large N to be fast but accurate
mu_value = 0.2
steps = 1500  # Enough time to reach steady state
num_seeds = 10
n_cores = 6

# We sweep the average degree <k> to create the X-axis of our phase diagram
k_values = np.linspace(2, 12, 11)  # From sparse (near percolation) to dense

# Thresholds to detect the boundaries
threshold_abs = 2e-4  # Below this, it's the Absorbing Phase
threshold_act = 5e-3  # Above this, it's the Active Endemic Phase

boundary_griffiths = []
boundary_active = []

print(f"[INFO] Sweeping average degree <k> from {k_values[0]} to {k_values[-1]}...")

for k_avg in k_values:
    print(f"\n       [~] Analyzing network with <k> = {k_avg:.1f}...")

    # Generate ER network
    num_edges = int((N * k_avg) / 2)
    rows = np.random.randint(0, N, size=num_edges)
    cols = np.random.randint(0, N, size=num_edges)
    data = np.ones(num_edges)

    matrix_adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    matrix_adj = matrix_adj + matrix_adj.transpose()
    matrix_adj.setdiag(0)
    matrix_adj.eliminate_zeros()
    matrix_adj.data = np.ones_like(matrix_adj.data)

    # The theoretical mean-field critical beta is mu / <k>
    # Since our max is lambda_max, the mean is lambda_max / 2.
    # Therefore, the transition should occur around lambda_max = 2 * mu / <k>
    center_lambda = 2 * mu_value / k_avg

    # Create a dynamic window around the expected transition
    lambda_grid = np.linspace(center_lambda * 0.4, center_lambda * 1.6, 25)

    # Run in parallel for the current <k>
    steady_states = Parallel(n_jobs=n_cores)(
        delayed(get_steady_state)(
            matrix_adj, l_max, mu_value, steps, num_seeds, 42, N
        )
        for l_max in lambda_grid
    )

    # Find the boundaries using the empirical steady states
    try:
        lam_g = next(l for l, rho in zip(lambda_grid, steady_states) if rho > threshold_abs)
    except StopIteration:
        lam_g = lambda_grid[-1]

    try:
        lam_a = next(l for l, rho in zip(lambda_grid, steady_states) if rho > threshold_act)
    except StopIteration:
        lam_a = lambda_grid[-1]

    boundary_griffiths.append(lam_g)
    boundary_active.append(lam_a)

    print(f"           -> Absorbing-Griffiths boundary: {lam_g:.4f}")
    print(f"           -> Griffiths-Active boundary: {lam_a:.4f}")

# ---------------------------------------------------------
# Plotting the 2D Phase Diagram
# ---------------------------------------------------------
print("\n[INFO] Generating 2D Phase Diagram...")

plt.figure(figsize=(10, 7))

# Plot the boundary lines
plt.plot(k_values, boundary_griffiths, 'o-', color='purple', linewidth=2, label='Extinction Boundary')
plt.plot(k_values, boundary_active, 's-', color='orange', linewidth=2, label='Endemic Boundary')

# Fill the regions (The core of the 2D phase diagram)
# 1. Absorbing Phase (Below the Griffiths boundary)
plt.fill_between(k_values, 0, boundary_griffiths, color='red', alpha=0.15)
plt.text(7, np.mean(boundary_griffiths) * 0.5, 'Absorbing Phase\n(Exponential Decay)',
         color='darkred', ha='center', va='center', fontsize=12, fontweight='bold')

# 2. Griffiths Phase (Between the boundaries)
plt.fill_between(k_values, boundary_griffiths, boundary_active, color='orange', alpha=0.3)
plt.text(7, np.mean(boundary_griffiths) + (np.mean(boundary_active) - np.mean(boundary_griffiths)) * 0.5,
         'Griffiths Phase\n(Algebraic Decay)',
         color='darkorange', ha='center', va='center', fontsize=12, fontweight='bold')

# 3. Active Phase (Above the Active boundary)
plt.fill_between(k_values, boundary_active, max(boundary_active) * 1.5, color='green', alpha=0.15)
plt.text(7, np.mean(boundary_active) * 1.2, 'Active Phase\n(Endemic State)',
         color='darkgreen', ha='center', va='center', fontsize=12, fontweight='bold')

# Formatting
plt.xlabel(r'Network Average Degree $\langle k \rangle$')
plt.ylabel(r'Maximum intrinsic infection rate $\lambda_{max}$')
plt.title('SIS Model 2D Phase Diagram with Quenched Disorder')

# Set Y-axis limit dynamically so the plot looks tight
plt.ylim(0, max(boundary_active) * 1.4)
plt.xlim(min(k_values), max(k_values))

plt.legend(loc='upper right', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.5)

plt_path = os.path.join(output_dir, "2D_Phase_Diagram.png")
plt.savefig(plt_path, dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print(f"[SUCCESS] 2D Phase Diagram generated and saved.")
print("=" * 60)