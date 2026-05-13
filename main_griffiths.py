import os
import random
from datetime import datetime
from typing import Any

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from griffiths_simulator import SISGriffithsSimulator

def find_phase_boundaries(lambda_values: np.ndarray, rho_steady: list, threshold_abs: float = 5e-5,
                          threshold_act: float = 5e-3) -> tuple:
    """
    Find the lambda_max boundaries for the absorbing, Griffiths, and active phases based on steady-state prevalence.

    :param lambda_values: numpy array, the swept lambda_max values
    :param rho_steady: list, the stationary infected fraction for each lambda
    :param threshold_abs: float, prevalence threshold below which the system is in the absorbing phase
    :param threshold_act: float, prevalence threshold above which the system is in the active phase
    :return: tuple, (lambda_start_griffiths, lambda_start_active)
    """
    try:
        # Find the first lambda where rho exceeds the absorbing threshold, marking the start of the Griffiths phase
        idx_griffiths = next(i for i, rho in enumerate(rho_steady) if rho > threshold_abs)
        lam_griffiths = lambda_values[idx_griffiths]
    except StopIteration:
        lam_griffiths = lambda_values[-1]

    try:
        # Find the first lambda where rho exceeds the active threshold, marking the start of the active phase
        idx_active = next(i for i, rho in enumerate(rho_steady) if rho > threshold_act)
        lam_active = lambda_values[idx_active]
    except StopIteration:
        lam_active = lambda_values[-1]

    return lam_griffiths, lam_active

def run_griffiths_decay(lambda_max: float, matrix_adj: Any, mu: float, steps: int,
                        num_seeds: int, seed: int, N: int) -> np.ndarray:
    """
    Execute multiple simulation runs for a specific lambda_max to compute the ensemble average decay.

    :param lambda_max: float, maximum possible infection rate for the uniform distribution
    :param matrix_adj: numpy array or sparse matrix (N, N), adjacency matrix of the graph
    :param mu: float, recovery rate
    :param steps: int, total number of time steps to simulate
    :param num_seeds: int, number of independent runs to average
    :param seed: int, base random seed for reproducibility
    :param N: int, number of nodes in the network
    :return: numpy array (steps,), ensemble-averaged time series of the infected fraction
    """
    temp_histories = []

    for s in range(num_seeds):
        current_seed = seed + s
        np.random.seed(current_seed)
        random.seed(current_seed)

        # Generate quenched disorder: each node gets a fixed random beta
        # Beta_i ~ U(0, lambda_max)
        betas = np.random.uniform(0, lambda_max, size=N)

        local_sim = SISGriffithsSimulator()
        # Start fully infected (initial_fraction=1.0) to observe the decay
        rho_history = local_sim.run(matrix_adj, betas, mu, steps, initial_fraction=1.0)
        temp_histories.append(rho_history)

    # Average the temporal decay across all seeds
    return np.mean(temp_histories, axis=0)


print("=" * 60)
print("       SIS Model - Griffiths Phase Exploration")
print("=" * 60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("results_griffiths", f"run_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Output directory created at: {output_dir}")

# Parameters specifically tuned for Griffiths Phase observation
N_values = [1000, 5000, 10000, 20000, 30000, 50000, 75000, 100000, 200000, 275000]
N_time_series = N_values[-1]
k_avg = 10
mu_value = 0.2
steps = 2000  # Long time needed to see the algebraic decay clearly
num_seeds = 25  # Crucial to average out the quenched disorder noise
seed = 42
n_cores = 6

# Lambda max serves as our control parameter.
# We sweep it to find the active, absorbing, and Griffiths phases.
lambda_values = np.linspace(0.008, 0.046, 16)

# Target lambda specifically for the Finite-Size Scaling analysis across all N
target_lambda = 0.019

print("[INFO] Starting simulation loops...")

results_fss = {}
results_lambda_sweep = []

for N in N_values:
    print(f"\n[INFO] Simulating for system size N={N}...")
    print(f"       [~] Generating Erdős-Rényi graph (N={N}, <k>={k_avg}) using sparse matrices...")

    # Calculate total undirected edges needed
    num_edges = int((N * k_avg) / 2)

    # Generate random edges (sampling with replacement is valid for large sparse graphs)
    rows = np.random.randint(0, N, size=num_edges)
    cols = np.random.randint(0, N, size=num_edges)
    data = np.ones(num_edges)

    # Build the sparse matrix in COO format, then convert to CSR for fast math
    matrix_adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    # Make the directed random graph undirected to simulate symmetric contacts
    matrix_adj = matrix_adj + matrix_adj.transpose()

    # Remove self-loops (nodes connected to themselves)
    matrix_adj.setdiag(0)
    matrix_adj.eliminate_zeros()

    # Ensure all edge weights are exactly 1, removing overlaps
    matrix_adj.data = np.ones_like(matrix_adj.data)

    # 1. Execute the fixed lambda_max simulation for Finite-Size Scaling analysis
    print(f"       --> Simulating decay for fixed lambda_max = {target_lambda:.3f}...")
    decay_curve = run_griffiths_decay(target_lambda, matrix_adj, mu_value, steps, num_seeds, seed, N)
    results_fss[N] = decay_curve

    # 2. Execute the full lambda_max sweep only for the largest network
    if N == N_time_series:
        print(f"       --> Simulating full lambda sweep for N={N}...")
        results_lambda_sweep = Parallel(n_jobs=n_cores)(
            delayed(run_griffiths_decay)(
                l_max, matrix_adj, mu_value, steps, num_seeds, seed, N
            )
            for l_max in lambda_values
        )

# ---------------------------------------------------------
# Plot 1: Finite-Size Cutoff in the Griffiths Phase (Log-Log)
# ---------------------------------------------------------
print("\n[INFO] Plotting Finite-Size Cutoff (Log-Log)...")

fig, ax = plt.subplots(figsize=(10, 8))
cmap_fss = plt.get_cmap('viridis')
colors_fss = cmap_fss(np.linspace(0, 0.9, len(N_values)))

for idx, n_val in enumerate(N_values):
    rho_data = results_fss[n_val]
    t_array = np.arange(1, steps + 1)

    # Ensure strict positivity for log-log plotting
    pos_indices = rho_data > 0

    ax.loglog(t_array[pos_indices], rho_data[pos_indices],
              color=colors_fss[idx], label=f'N = {n_val}')

ax.set_xlabel('Time steps ($t$)')
ax.set_ylabel(r'Infected fraction $\rho(t)$')
ax.set_title(rf'Finite-Size Cutoff in the Griffiths Phase ($\lambda_{{max}}={target_lambda}$)')
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

# Clean background with inward ticks on all sides (APS journal style)
ax.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)

plt_path_fss = os.path.join(output_dir, "griffiths_finite_size_cutoff.png")
plt.savefig(plt_path_fss, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Plot 2: Algebraic Decay in the Griffiths Phase (Log-Log)
# ---------------------------------------------------------
print("[INFO] Plotting algebraic decay sweep (Log-Log)...")

from scipy import stats

fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.get_cmap('plasma')
colors = cmap(np.linspace(0, 0.9, len(lambda_values)))

# Create inset axes in the bottom-left corner
# [x0, y0, width, height] in axes-relative coordinates
axins = ax.inset_axes([0.09, 0.08, 0.35, 0.35])

for idx, (l_max, rho_history) in enumerate(zip(lambda_values, results_lambda_sweep)):
    t_array = np.arange(1, steps + 1)

    # Ensure strict positivity for log-log plotting
    pos_indices = rho_history > 0
    t_pos = t_array[pos_indices]
    rho_pos = rho_history[pos_indices]

    # Main plot: log-log scale for algebraic decay
    ax.loglog(t_pos, rho_pos,
              color=colors[idx], label=rf'$\lambda_{{max}} = {l_max:.3f}$')

    # Inset plot: semi-log-y scale to highlight exponential decay
    axins.semilogy(t_pos, rho_pos, color=colors[idx])

ax.set_xlabel('Time steps ($t$)')
ax.set_ylabel(r'Infected fraction $\rho(t)$')
ax.set_title(f'Griffiths Phase Algebraic Decay (N={N_time_series})')
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

# Clean background with inward ticks on all sides (main plot)
ax.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)

# Format the inset plot to match the style
axins.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)
axins.set_xlabel('$t$', fontsize=10)
axins.set_ylabel(r'$\rho(t)$', fontsize=10)
axins.set_title('Semi-log scale', fontsize=10)

# =========================================================
# SAVE 1: The pristine base plot
# =========================================================
plt_path_base = os.path.join(output_dir, f"griffiths_decay_loglog_N{N_time_series}_base.png")
plt.savefig(plt_path_base, dpi=300, bbox_inches='tight')
print(f"       -> Saved base plot: {plt_path_base}")

# =========================================================
# ADD THEORETICAL FITS TO THE OPEN CANVAS
# =========================================================
for idx, (l_max, rho_history) in enumerate(zip(lambda_values, results_lambda_sweep)):

    # Select two specific curves inside the Griffiths phase to fit
    if abs(l_max - 0.026) < 0.0015 or abs(l_max - 0.028) < 0.0015:
        t_array = np.arange(1, steps + 1)

        # Define the temporal window to fit (avoids initial transient and finite-size tail)
        fit_start = 40
        fit_end = 300

        valid_fit = (rho_history > 0) & (t_array > fit_start) & (t_array < fit_end)

        if np.any(valid_fit):
            t_fit = t_array[valid_fit]
            rho_fit = rho_history[valid_fit]

            # Linear regression in log-log space
            log_t = np.log(t_fit)
            log_rho = np.log(rho_fit)
            slope, intercept, _, _, _ = stats.linregress(log_t, log_rho)

            # Generate the theoretical line points
            fit_line = np.exp(intercept) * (t_fit ** slope)

            # Draw the dashed line on the main axis
            ax.loglog(t_fit, fit_line, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

            # Add the exponent text annotation next to the line
            ax.text(t_fit[-1] * 1.1, fit_line[-1], rf'$\theta \approx {-slope:.2f}$',
                    color=colors[idx], fontsize=11, fontweight='bold', verticalalignment='center')

# =========================================================
# SAVE 2: The plot with theoretical fit lines
# =========================================================
plt_path_fits = os.path.join(output_dir, f"griffiths_decay_loglog_N{N_time_series}_with_fits.png")
plt.savefig(plt_path_fits, dpi=300, bbox_inches='tight')
print(f"       -> Saved plot with fits: {plt_path_fits}")

# Finally, close the figure to free up memory
plt.close()

# ---------------------------------------------------------
# Plot 3: Time Evolution in Linear Scale
# ---------------------------------------------------------
print("\n[INFO] Plotting time evolution (Linear Scale)...")

fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.get_cmap('plasma')
colors = cmap(np.linspace(0, 0.9, len(lambda_values)))

for idx, (l_max, rho_history) in enumerate(zip(lambda_values, results_lambda_sweep)):
    t_array = np.arange(1, steps + 1)

    # Standard linear plot
    ax.plot(t_array, rho_history, color=colors[idx], label=rf'$\lambda_{{max}} = {l_max:.3f}$')

ax.set_xlabel('Time steps ($t$)')
ax.set_ylabel(r'Infected fraction $\rho(t)$')
ax.set_title(f'Griffiths Phase Time Evolution (N={N_time_series})')
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

# Clean background with inward ticks on all sides (APS journal style)
ax.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)
ax.grid(True, linestyle="--", alpha=0.3)

plt_path_linear = os.path.join(output_dir, f"griffiths_decay_linear_N{N_time_series}.png")
plt.savefig(plt_path_linear, dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print(f"[SUCCESS] Griffiths phase simulation finished.")
print("=" * 60)