import os
import random
from datetime import datetime
from typing import Any

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy import floating
from scipy import stats
from sis_simulator import compute_susceptibility, SISSimulator

# Only for parallel simulation
from joblib import Parallel, delayed

def calc_dbmf(graph: nx.Graph, mu: float) -> floating[Any]:
    """
    Degree-Based Mean-Field (DBMF) approximation for the critical infection rate beta_c in the SIS model on a network.
    It uses the heterogeneity of the degree distribution to estimate the epidemic threshold.
    :param graph: networkx graph
    :param mu: float, recovery rate
    :return: float, estimated critical infection rate beta_c
    """
    degrees = [d for n, d in graph.degree()]
    k_mean = np.mean(degrees)
    k2_mean = np.mean(np.array(degrees) ** 2)
    return mu * k_mean / k2_mean


def run_single_beta(b: float, matrix_adj: Any, mu: float, steps: int, transient: int, initial_fraction: float,
                    num_seeds: int, seed: int, N: int, is_time_series_n: bool) -> tuple:
    """
    Function to be executed in parallel for a single beta value.

    :param b: float, infection rate for this run
    :param matrix_adj: numpy array (N, N), adjacency matrix of the graph
    :param mu: float, recovery rate
    :param steps: int, total number of time steps to simulate
    :param transient: int, number of initial steps to discard for steady-state analysis
    :param initial_fraction: float, initial fraction of infected nodes
    :param num_seeds: int, number of independent runs to average for this beta
    :param seed: int, base random seed for reproducibility
    :param N: int, number of nodes in the network (used for susceptibility calculation)
    :param is_time_series_n: bool, whether this N is the one for which we want to save the time series data for plotting
    :return: tuple (rho_mean, chi_mean, history_to_save) where:
             - rho_mean: float, average infected fraction in steady state for this beta
             - chi_mean: float, average susceptibility for this beta
             - history_to_save: numpy array or None, the averaged time series of infected fraction if is_time_series_n
             is True, otherwise None
    """

    # Temp lists for tracking data
    rho_means = []
    chi_values = []
    history_to_save = None
    temp_histories = []

    # Execute simulations for the current beta
    for s in range(num_seeds):
        current_seed = seed + s
        np.random.seed(current_seed)
        random.seed(current_seed)

        # Execute simulation using a local simulator to avoid thread safety issues
        local_sim = SISSimulator()
        rho_full_history = local_sim.run(matrix_adj, b, mu, steps, transient, initial_fraction)

        # Slice the array to get only the steady-state for statistics
        rho_steady_state = rho_full_history[transient:]

        # Calculate and save the stats for this specific run
        rho_means.append(np.mean(rho_steady_state))
        chi_values.append(compute_susceptibility(rho_steady_state, N))

        # Cache history only if it's the chosen N for time series
        if is_time_series_n:
            temp_histories.append(rho_full_history)

    # Save the averaged time series (if applicable)
    if is_time_series_n:
        history_to_save = np.mean(temp_histories, axis=0)

    return np.mean(rho_means), np.mean(chi_values), history_to_save


print("=" * 60)
print("       SIS Model Simulation on Complex Networks")
print("=" * 60)

# Setup output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("results", f"run_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Output directory created at: {output_dir}")

# Execution options
use_parallel = False
n_cores = 12

print(f"[INFO] Execution mode: {f'PARALLEL ({n_cores} cores)' if use_parallel else 'SEQUENTIAL'}")

# Parameters
# List of N values for Finite-Size Scaling analysis
N_values = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000]
# Specify which N should be used for the time series plots (usually the largest)
N_time_series = N_values[-1]
k_avg = 10   # Average degree <k> = 5
mu_value = 0.2
beta_values = np.linspace(0.002, 0.08, 50)
steps = 1000
transient = 250
initial_fraction = 0.1

# Number of betas for the rho vs time plot
num_time_series_plots = 10
ts_indexes = np.linspace(0, len(beta_values) - 1, num_time_series_plots, dtype=int)

# Number of independent simulations to average per beta
num_seeds = 5
seed = 42

visual_axis = ""
for i in range(len(beta_values)):
    if i in ts_indexes:
        visual_axis += "█"
    else:
        visual_axis += "-"

print(f"[INFO] Initializing parameters: Ns={N_values}, <k>={k_avg}, steps={steps}, transient={transient}")
print(f"[INFO] Initial infected fraction: {initial_fraction*100}%")
print(f"[INFO] Rates: mu={mu_value}, beta range=({beta_values[0]:.2f} to {beta_values[-1]:.2f}) with "
      f"{len(beta_values)} points")
print(f"[INFO] Ensemble averaging over {num_seeds} runs per beta.")
print(f"[INFO] Tracking time evolution for N={N_time_series} at {num_time_series_plots} evenly distributed betas.")
print(f"[INFO] Axis: {beta_values[0]:.3f} [{visual_axis}] {beta_values[-1]:.3f}")
print("[INFO] Generating the 3 networks (Erdős-Rényi, Watts-Strogatz, Barabási-Albert)...")

network_names = ["Erdős-Rényi", "Watts-Strogatz", "Barabási-Albert"]

# Dicts for the results (Nested dictionaries: results_rho[network_name][N])
results_rho = {name: {N: [] for N in N_values} for name in network_names}
results_chi = {name: {N: [] for N in N_values} for name in network_names}

# Theoretical beta values storage for two approximation
# 1. Mean-Field Approximation (MFA): Assumes a homogeneous network
beta_c_mfa = mu_value / k_avg
# 2. Degree-Based Mean-Field (DBMF): Accounts for degree heterogeneity
beta_c_dbmf_dict = {name: {} for name in network_names}

# Dict to cache the full time series for rho vs time plot (only for N_time_series)
saved_time_series = {name: {} for name in network_names}

# The simulator class
simulator = SISSimulator()

print("[INFO] Starting simulation loops...")

for N in N_values:
    print("\n" + "=" * 60)
    print(f"[>>>] SIMULATING FOR SYSTEM SIZE N={N}")
    print("=" * 60)

    # Generate the networks for the current N
    networks = {
        "Erdős-Rényi": nx.erdos_renyi_graph(N, k_avg / N),
        "Watts-Strogatz": nx.watts_strogatz_graph(N, k_avg, 0.1),
        "Barabási-Albert": nx.barabasi_albert_graph(N, k_avg // 2)
    }

    beta_c_dbmf = {net_name: calc_dbmf(G, mu_value) for net_name, G in networks.items()}

    for name, G in networks.items():
        print(f"\n       [!] Simulating network: {name}...")

        # Calculate and store the Degree-Based Mean-Field (DBMF) critical beta for this N
        beta_c_dbmf_dict[name][N] = calc_dbmf(G, mu_value)

        # Get the adjacency matrix
        matrix_adj = nx.adjacency_matrix(G)

        if use_parallel:
            # Parallel execution
            results = Parallel(n_jobs=n_cores)(
                delayed(run_single_beta)(
                    b, matrix_adj, mu_value, steps, transient, initial_fraction,
                    num_seeds, seed, N, (N == N_time_series and b_idx in ts_indexes)
                )
                for b_idx, b in enumerate(beta_values)
            )
        else:
            # Sequential execution (with progress printouts)
            results = []
            for b_idx, b in enumerate(beta_values):
                print(f"       --> Running beta={b:.4f} (Averaging {num_seeds} runs)...", end="\r", flush=True)

                res = run_single_beta(
                    b, matrix_adj, mu_value, steps, transient, initial_fraction,
                    num_seeds, seed, N, (N == N_time_series and b_idx in ts_indexes)
                )
                results.append(res)

        # Process the results is the same for both methods
        for b_idx, (r_mean, c_mean, hist) in enumerate(results):
            results_rho[name][N].append(r_mean)
            results_chi[name][N].append(c_mean)

            if hist is not None:
                saved_time_series[name][beta_values[b_idx]] = hist


        print(f"       [+] Finished beta sweep for {name} (N={N}). Theoretical beta_c (DBMF): "
              f"{beta_c_dbmf_dict[name][N]:.4f}")

        # Save raw data for this network and this N
        data_to_save = np.column_stack((beta_values, results_rho[name][N], results_chi[name][N]))
        safe_name = (name.replace("ő", "o").replace("é", "e").replace("á", "a")
                     .replace("-", "_"))
        dat_filepath = os.path.join(output_dir, f"{safe_name}_N{N}_results.dat")
        np.savetxt(dat_filepath, data_to_save, header="beta rho_mean susceptibility", fmt="%.6f")

print("\n[INFO] All simulations completed successfully.")
print("[INFO] Generating and saving plots...")

# To store the critical beta for the log-log plots: beta_c_dict[network_name][N]
beta_c_dict = {name: {} for name in network_names}

for name in network_names:
    safe_name = (name.replace("ő", "o").replace("é", "e").replace("á", "a")
                 .replace("-", "_"))

    # ---------------------------------------------------------
    # Plot 1: Prevalence (Phase Transition) for multiple N
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for N in N_values:
        plt.plot(beta_values, results_rho[name][N], marker='o', markersize=4, label=f'Data N={N}')

    # Add Theoretical Values for this specific network (using the largest N for DBMF reference)
    plt.axvline(x=beta_c_mfa, color='black', linestyle=':', label='MFA (Homogeneous)')
    plt.axvline(x=beta_c_dbmf_dict[name][N_time_series], color='gray', linestyle='-.', label=f'DBMF ({name})')

    plt.xlabel(r'Infection rate $\beta$')
    plt.ylabel(r'Stationary infected fraction $\langle \rho \rangle$')
    plt.title(f'SIS Phase Transition - {name}')
    # Place legend outside the plot to avoid overlapping
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"1_{safe_name}_phase_transition.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Susceptibility (Finding beta_c) for multiple N
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for N in N_values:
        line = plt.plot(beta_values, results_chi[name][N], marker='s', markersize=4, label=f'Data N={N}')[0]
        color_line = line.get_color()

        # Find the index of the maximum susceptibility
        max_idx = np.argmax(results_chi[name][N])
        beta_c_dict[name][N] = beta_values[max_idx]

        # Mark the empirical peak
        plt.axvline(x=beta_c_dict[name][N], color=color_line, linestyle='--', alpha=0.5, label=f'Peak N={N}')
        print(f"[INFO] Estimated beta_c for {name} (N={N}): {beta_c_dict[name][N]:.4f}")

    # Add Theoretical Values for this specific network
    plt.axvline(x=beta_c_mfa, color='black', linestyle=':', label='MFA (Homogeneous)')
    plt.axvline(x=beta_c_dbmf_dict[name][N_time_series], color='gray', linestyle='-.', label=f'DBMF ({name})')

    plt.xlabel(r'Infection rate $\beta$')
    plt.ylabel(r'Susceptibility $\chi$')
    plt.title(f'Finite-Size Scaling (Susceptibility) - {name}')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")  # ncol=2 for a cleaner look with many Ns
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"2_{safe_name}_susceptibility.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Plot 3: Critical Exponent (Log-Log) for multiple N
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for N in N_values:
        beta_c = beta_c_dict[name][N]
        valid_indices = beta_values > beta_c

        x_data = beta_values[valid_indices] - beta_c
        y_data = np.array(results_rho[name][N])[valid_indices]

        # Ensure strict positivity
        pos_indices = y_data > 0
        x_data = x_data[pos_indices]
        y_data = y_data[pos_indices]

        if len(x_data) > 1:
            line = plt.loglog(x_data, y_data, marker='^', linestyle='', label=f'N={N} data')[0]
            color_line = line.get_color()

            log_x = np.log(x_data)
            log_y = np.log(y_data)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

            fit_y = np.exp(intercept) * (x_data ** slope)
            plt.loglog(x_data, fit_y, linestyle='-', color=color_line, label=f'N={N} fit (exp: {slope:.2f})')

    plt.xlabel(r'$(\beta - \beta_c)$')
    plt.ylabel(r'$\langle \rho \rangle$')
    plt.title(f'Critical Scaling - {name}')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"3_{safe_name}_critical_scaling.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# Plot 4: Time Evolution of Infected Fraction (Cached)
# ---------------------------------------------------------
print("[INFO] Generating Plot 4: Time Evolution from cached data...")

for name in network_names:
    safe_name = (name.replace("ő", "o").replace("é", "e").replace("á", "a")
                 .replace("-", "_"))
    cached_data = saved_time_series[name]

    if not cached_data:
        continue

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 0.9, len(cached_data)))

    for (b, history), color in zip(cached_data.items(), colors):
        plt.plot(range(len(history)), history, label=rf'$\beta={b:.4f}$', color=color, alpha=0.8)

    plt.axvline(x=transient, color='red', linestyle='--', alpha=0.7, label=f'End of Transient ({transient})')

    plt.xlabel('Time steps ($t$)')
    plt.ylabel(r'Infected fraction ($\rho$)')
    plt.title(f'Time Evolution (N={N_time_series}) - {name}')
    plt.legend(loc='center right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"4_{safe_name}_time_evolution.png"), dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================================
# NETWORK COMPARISON PLOTS (Fixed at N = N_time_series)
# =========================================================================
print(f"[INFO] Generating Network Comparison Plots for N={N_time_series}...")

comparison_color_map = {
    "Erdős-Rényi": "blue",
    "Watts-Strogatz": "green",
    "Barabási-Albert": "red"
}

# ---------------------------------------------------------
# Plot 5: Comparison Prevalence (Phase Transition)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# Define a custom sorting order for the network types in legend
sorted_names_for_legend = ["Erdős-Rényi", "Watts-Strogatz", "Barabási-Albert"]

# Plot lines and store handles for sorting legend later
plot_handles = {}

for name in sorted_names_for_legend:
    if name not in networks: continue  # Safety check

    # Extract color from semantic map
    net_color = comparison_color_map.get(name, "black")

    # Explicitly use [N_time_series] data
    line = plt.plot(beta_values, results_rho[name][N_time_series],
                    marker='o', markersize=4, color=net_color,
                    label=f'Data: {name}')[0]

    # Store handle using a tuple key (Network Name, 'data')
    plot_handles[(name, 'data')] = line

    # Add corresponding DBMF for this network with same color but different style
    dbmf_line = plt.axvline(x=beta_c_dbmf_dict[name][N_time_series],
                            color=net_color, linestyle='-.', alpha=0.7,
                            label=f'DBMF: {name}')

    # Store handle
    plot_handles[(name, 'dbmf')] = dbmf_line

# Add the universal MFA line (stays black)
mfa_line = plt.axvline(x=beta_c_mfa, color='black', linestyle=':', label='MFA (Homogeneous)')

# Combine handles for legend sorting
all_handles = [plot_handles[(n, 'data')] for n in sorted_names_for_legend if (n, 'data') in plot_handles] + \
              [plot_handles[(n, 'dbmf')] for n in sorted_names_for_legend if (n, 'dbmf') in plot_handles] + \
              [mfa_line]

plt.xlabel(r'Infection rate $\beta$')
plt.ylabel(r'Stationary infected fraction $\langle \rho \rangle$')
plt.title(f'SIS Epidemic Phase Transition Comparison (N={N_time_series})')

# Create sorted legend placed outside
plt.legend(handles=all_handles, bbox_to_anchor=(1.04, 1), loc="upper left")
plt.grid(True, alpha=0.3)
plt_path_5 = os.path.join(output_dir, f"5_comparison_phase_transition_N{N_time_series}.png")
plt.savefig(plt_path_5, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Plot 6: Comparison Susceptibility (Finding beta_c)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
beta_c_comp_dict = {}

# Reset plot handles for this specific graph
plot_handles = {}

for name in sorted_names_for_legend:
    if name not in networks: continue # Safety check

    net_color = comparison_color_map.get(name, "black")

    # Plot empirical data
    line = plt.plot(beta_values, results_chi[name][N_time_series],
                    marker='s', markersize=4, color=net_color,
                    label=f'Data: {name}')[0]
    plot_handles[(name, 'data')] = line

    # Find and mark empirical peak
    max_idx = np.argmax(results_chi[name][N_time_series])
    beta_c_comp_dict[name] = beta_values[max_idx]
    peak_line = plt.axvline(x=beta_c_comp_dict[name], color=net_color,
                            linestyle='--', alpha=0.5, label=f'Peak: {name}')
    plot_handles[(name, 'peak')] = peak_line

    # Add theoretical DBMF
    dbmf_line = plt.axvline(x=beta_c_dbmf_dict[name][N_time_series],
                            color=net_color, linestyle='-.', alpha=0.7,
                            label=f'DBMF: {name}')
    plot_handles[(name, 'dbmf')] = dbmf_line

# Add universal MFA
mfa_line = plt.axvline(x=beta_c_mfa, color='black', linestyle=':', label='MFA (Homogeneous)')

# Group and sort handles: Data, then Peak, then Theory for each network
grouped_handles = []
for n in sorted_names_for_legend:
    if (n, 'data') in plot_handles: grouped_handles.append(plot_handles[(n, 'data')])
    if (n, 'peak') in plot_handles: grouped_handles.append(plot_handles[(n, 'peak')])
    if (n, 'dbmf') in plot_handles: grouped_handles.append(plot_handles[(n, 'dbmf')])
grouped_handles.append(mfa_line)

plt.xlabel(r'Infection rate $\beta$')
plt.ylabel(r'Susceptibility $\chi$')
plt.title(f'Susceptibility Peak Comparison (N={N_time_series})')

# Organized legend
plt.legend(handles=grouped_handles, bbox_to_anchor=(1.04, 1), loc="upper left")
plt.grid(True, alpha=0.3)
plt_path_6 = os.path.join(output_dir, f"6_comparison_susceptibility_N{N_time_series}.png")
plt.savefig(plt_path_6, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Plot 7: Comparison Finite-Size Scaling (Log-Log)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

for name in network_names:
    net_color = comparison_color_map.get(name, "black")
    beta_c = beta_c_comp_dict[name]

    # Filter data: active phase (beta > beta_c)
    valid_indices = beta_values > beta_c
    x_data = beta_values[valid_indices] - beta_c
    y_data = np.array(results_rho[name][N_time_series])[valid_indices]

    # Ensure strict positivity
    pos_indices = y_data > 0
    x_data = x_data[pos_indices]
    y_data = y_data[pos_indices]

    if len(x_data) > 1:
        # Scatter plot in log-log scale with semantic color
        plt.loglog(x_data, y_data, marker='^', linestyle='',
                   color=net_color, label=f'Data: {name}')

        # Linear regression and fit line (same color, solid style)
        log_x = np.log(x_data)
        log_y = np.log(y_data)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        fit_y = np.exp(intercept) * (x_data ** slope)
        plt.loglog(x_data, fit_y, linestyle='-', color=net_color,
                   label=f'Fit {name} (exp: {slope:.2f})')

plt.xlabel(r'$(\beta - \beta_c)$')
plt.ylabel(r'$\langle \rho \rangle$')
plt.title(f'Critical Scaling Comparison (N={N_time_series})')
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.grid(True, which="both", ls="--", alpha=0.3)
plt_path_7 = os.path.join(output_dir, f"7_comparison_critical_scaling_N{N_time_series}.png")
plt.savefig(plt_path_7, dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print(f"[SUCCESS] All tasks finished. Check your data at: ./{output_dir}/")
print("=" * 60)