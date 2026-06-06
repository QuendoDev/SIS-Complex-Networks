import json
import os
import random
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# Only for parallel simulation
from joblib import Parallel, delayed
from numpy import floating
from scipy import stats

from src.simulators.sis_simulator import compute_susceptibility, SISSimulator


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


def run_single_beta(b: float, matrix_adj: Any, mu: float, steps: int, transient: int,
                    initial_fraction: float, num_seeds: int, seed: int, N: int, is_time_series_n: bool) -> tuple:
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
    :return: tuple (rho_mean, rho_std, chi_mean, chi_std, history_to_save)
    """

    rho_means = []
    chi_values = []
    history_to_save = None
    temp_histories = []

    for s in range(num_seeds):
        current_seed = seed + s
        np.random.seed(current_seed)
        random.seed(current_seed)

        local_sim = SISSimulator()
        rho_full_history = local_sim.run(matrix_adj, b, mu, steps, transient, initial_fraction)

        rho_steady_state = rho_full_history[transient:]

        rho_means.append(np.mean(rho_steady_state))
        chi_values.append(compute_susceptibility(rho_steady_state, N))

        if is_time_series_n:
            temp_histories.append(rho_full_history)

    if is_time_series_n:
        history_to_save = np.mean(temp_histories, axis=0)

    return np.mean(rho_means), np.std(rho_means), np.mean(chi_values), np.std(chi_values), history_to_save


def run_dense_grid(beta_c: float, tolerance: float, points: int, matrix_adj: Any, mu: float, steps: int,
                   transient: int, initial_fraction: float, num_seeds: int, seed: int, N: int,
                   n_cores: int) -> tuple:
    """
    Generate and simulate a dense grid of beta values strictly just above the critical point to fit critical exponents.

    :param beta_c: float, previously calculated critical infection rate
    :param tolerance: float, width of the dense window above beta_c
    :param points: int, number of points in the dense grid
    :param matrix_adj: numpy array (N, N), adjacency matrix
    :param mu: float, recovery rate
    :param steps: int, total steps
    :param transient: int, steps to discard
    :param initial_fraction: float, initial infected fraction
    :param num_seeds: int, runs to average
    :param seed: int, random seed
    :param N: int, number of nodes
    :param n_cores: int, cores for parallelization
    :return: tuple (dense_betas, dense_rho_means), arrays of beta values and their corresponding steady rho
    """
    dense_betas = np.linspace(beta_c + 1e-4, beta_c + tolerance, points)

    results = Parallel(n_jobs=n_cores, backend="threading")(
        delayed(run_single_beta)(
            b, matrix_adj, mu, steps, transient, initial_fraction, num_seeds, seed, N, False
        )
        for b in dense_betas
    )

    dense_rho_means = np.array([res[0] for res in results])
    return dense_betas, dense_rho_means


print("=" * 60)
print("       SIS Model Simulation on Complex Networks")
print("=" * 60)

# =========================================================================
# 1. USER INTERFACE & DIRECTORY SETUP
# =========================================================================
print("Select data generation mode:")
print("  [1] Use existing data from cache (Generate NOTHING, just plot)")
print("  [2] Generate ONLY dense grid data (Requires cached main data)")
print("  [3] Generate ALL data (Clears cache and runs full 100% simulation)")
user_input = input("[?] Enter your choice (1/2/3): ").strip()

# Set boolean flags based on user choice
generate_main = False
generate_dense = False

if user_input == '3':
    generate_main = True
    generate_dense = True
elif user_input == '2':
    generate_main = False
    generate_dense = True
else:  # Default to 1 (or any invalid input for safety)
    generate_main = False
    generate_dense = False

data_dir = os.path.join("../../results", "data_cache")
plots_dir = os.path.join("../../results", "plots")

# Directory and Cache management based on the flags
if generate_main:
    os.makedirs(data_dir, exist_ok=True)
    # Clear previous cache to avoid mixing old and new simulation data
    for f in os.listdir(data_dir):
        os.remove(os.path.join(data_dir, f))
    print(f"[INFO] Cleared cache. Generating ALL new data into: {data_dir}")
elif generate_dense:
    # Safety check: We cannot do a dense grid if we don't have the main beta_c calculated
    if not os.path.exists(data_dir) or not any(f.startswith("main_") for f in os.listdir(data_dir)):
        print("[ERROR] No cached main data found to build dense grid upon. Reverting to Generate ALL.")
        generate_main = True
        os.makedirs(data_dir, exist_ok=True)
    else:
        print(f"[INFO] Keeping main cache. Generating ONLY dense grid into: {data_dir}")
else:
    # Safety check: We can't plot if there is no data
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print("[ERROR] No cached data found. Defaulting to Generate ALL.")
        generate_main = True
        generate_dense = True
        os.makedirs(data_dir, exist_ok=True)
    else:
        print(f"[INFO] Loading ALL data from cache: {data_dir}")

os.makedirs(plots_dir, exist_ok=True)

# =========================================================================
# 2. PARAMETERS & INITIALIZATION
# =========================================================================
use_parallel = True
n_cores = 12

N_values = [2000, 10000, 50000, 100000, 500000]
N_time_series = N_values[-1]
k_avg = 10
mu_value = 0.2
beta_values = np.linspace(0.002, 0.05, 50)
steps = 1000
initial_fraction = 0.1

# Interactive prompt for the transient time to avoid hardcoding
print("\n[?] Simulation Settings:")
transient_input = input("  Enter the transient time to discard (press Enter for default 250): ").strip()

# If the user inputs a valid positive number, use it. Otherwise, fallback to default.
if transient_input.isdigit():
    transient = int(transient_input)
    print(f"  -> Transient time manually set to: {transient}")
else:
    transient = 250
    print(f"  -> Using default transient time: {transient}")

# Ask the user if the transient line should be drawn in the time evolution plots
plot_trans_input = (input("  Do you want to plot the transient line in the time evolution graphs? (y/n, default y): ")
                    .strip().lower())
plot_transient_line = False if plot_trans_input == 'n' else True
print(f"  -> Plot transient line: {plot_transient_line}")

num_time_series_plots = 10
ts_indexes = np.linspace(0, len(beta_values) - 1, num_time_series_plots, dtype=int)
num_seeds = 6
seed = 42

# Dense grid parameters for critical exponents (Log-Log)
dense_tolerance = 0.01
dense_points = 50

network_names = ["Erdős-Rényi", "Watts-Strogatz", "Barabási-Albert"]

# Universal MFA
beta_c_mfa = mu_value / k_avg

# Data Structures
results_rho = {name: {N: [] for N in N_values} for name in network_names}
results_rho_err = {name: {N: [] for N in N_values} for name in network_names}
results_chi = {name: {N: [] for N in N_values} for name in network_names}
results_chi_err = {name: {N: [] for N in N_values} for name in network_names}
beta_c_dbmf_dict = {name: {} for name in network_names}
saved_time_series = {name: {} for name in network_names}
beta_c_dict = {name: {} for name in network_names}
dense_results_rho = {name: {N: [] for N in N_values} for name in network_names}
dense_results_betas = {name: {N: [] for N in N_values} for name in network_names}


def get_safe_name(raw_name: str) -> str:
    """
    Sanitize network name for file saving.

    :param raw_name: str, original network name
    :return: str, sanitized name
    """
    return (raw_name.replace("ő", "o").replace("é", "e").replace("á", "a")
            .replace("-", "_"))


# =========================================================================
# 3. DATA GENERATION / LOADING
# =========================================================================

# --- PRE-LOAD DICTIONARIES IF NOT GENERATING MAIN DATA ---
# We need the calculated beta_c and DBMF to process the dense grid or plots
if not generate_main:
    print("[INFO] Loading main dictionaries from cache...")
    with open(os.path.join(data_dir, "beta_c_dict.json"), "r") as f:
        loaded_bc = json.load(f)
    with open(os.path.join(data_dir, "beta_c_dbmf.json"), "r") as f:
        loaded_dbmf = json.load(f)

    for name in network_names:
        safe_name = get_safe_name(name)
        beta_c_dict[name] = {int(k): float(v) for k, v in loaded_bc[safe_name].items()}
        beta_c_dbmf_dict[name] = {int(k): float(v) for k, v in loaded_dbmf[safe_name].items()}

# --- EXECUTE SIMULATIONS OR LOAD PARTIAL DATA ---
if generate_main or generate_dense:
    print(f"[INFO] Initializing parameters: Ns={N_values}, <k>={k_avg}, steps={steps}")
    print("[INFO] Starting simulation loops...")

    for N in N_values:
        print("\n" + "=" * 60)
        print(f"[>>>] PROCESSING SYSTEM SIZE N={N}")
        print("=" * 60)

        # Generate the networks dynamically to save RAM (overwritten each N)
        er_graph = nx.erdos_renyi_graph(N, k_avg / N)
        ws_graph = nx.watts_strogatz_graph(N, k_avg, 0.1)
        ba_graph = nx.barabasi_albert_graph(N, k_avg // 2)

        networks = {"Erdős-Rényi": er_graph, "Watts-Strogatz": ws_graph, "Barabási-Albert": ba_graph}

        for name, G in networks.items():
            safe_name = get_safe_name(name)
            print(f"\n       [!] Network: {name}...")

            matrix_adj = nx.adjacency_matrix(G)

            # --- PHASE A: MAIN GRID ---
            if generate_main:
                beta_c_dbmf_dict[name][N] = calc_dbmf(G, mu_value)

                results = Parallel(n_jobs=n_cores, backend="threading")(
                    delayed(run_single_beta)(
                        b, matrix_adj, mu_value, steps, transient, initial_fraction,
                        num_seeds, seed, N, (N == N_time_series and b_idx in ts_indexes)
                    )
                    for b_idx, b in enumerate(beta_values)
                )

                for b_idx, (r_mean, r_std, c_mean, c_std, hist) in enumerate(results):
                    results_rho[name][N].append(r_mean)
                    results_rho_err[name][N].append(r_std)
                    results_chi[name][N].append(c_mean)
                    results_chi_err[name][N].append(c_std)

                    if hist is not None:
                        saved_time_series[name][beta_values[b_idx]] = hist

                # Calculate empirical beta_c via susceptibility peak
                max_idx = np.argmax(results_chi[name][N])
                beta_c_dict[name][N] = beta_values[max_idx]
                print(f"       [+] beta_c estimated at: {beta_c_dict[name][N]:.4f}")

                main_data = np.column_stack((beta_values, results_rho[name][N], results_rho_err[name][N],
                                             results_chi[name][N], results_chi_err[name][N]))
                np.savetxt(os.path.join(data_dir, f"main_{safe_name}_N{N}.dat"), main_data)

                # Save time series if applicable
                if N == N_time_series:
                    for b_val, h_array in saved_time_series[name].items():
                        np.save(os.path.join(data_dir, f"hist_{safe_name}_b{b_val:.4f}.npy"), h_array)
            else:
                # If generating dense only, load main data so it's available for plotting
                main_data = np.loadtxt(os.path.join(data_dir, f"main_{safe_name}_N{N}.dat"))
                beta_values = main_data[:, 0]
                results_rho[name][N] = main_data[:, 1].tolist()
                results_rho_err[name][N] = main_data[:, 2].tolist()
                results_chi[name][N] = main_data[:, 3].tolist()
                results_chi_err[name][N] = main_data[:, 4].tolist()

            # --- PHASE B: DENSE GRID ---
            if generate_dense:
                print(f"       [~] Running dense grid for exponents ({dense_points} points)...")
                d_betas, d_rhos = run_dense_grid(
                    beta_c_dict[name][N], dense_tolerance, dense_points, matrix_adj, mu_value,
                    steps, transient, initial_fraction, num_seeds, seed, N, n_cores
                )
                dense_results_betas[name][N] = d_betas
                dense_results_rho[name][N] = d_rhos

                dense_data = np.column_stack((d_betas, d_rhos))
                np.savetxt(os.path.join(data_dir, f"dense_{safe_name}_N{N}.dat"), dense_data)
            else:
                # Fallback if somehow requested but not calculated
                dense_data = np.loadtxt(os.path.join(data_dir, f"dense_{safe_name}_N{N}.dat"))
                dense_results_betas[name][N] = dense_data[:, 0]
                dense_results_rho[name][N] = dense_data[:, 1]

    # Save dictionaries to JSON only if main data was freshly generated
    if generate_main:
        with open(os.path.join(data_dir, "beta_c_dict.json"), "w") as f:
            json.dump({get_safe_name(k): v for k, v in beta_c_dict.items()}, f)
        with open(os.path.join(data_dir, "beta_c_dbmf.json"), "w") as f:
            json.dump({get_safe_name(k): v for k, v in beta_c_dbmf_dict.items()}, f)

# --- LOAD FULL CACHE (IF GENERATE NOTHING) ---
if not generate_main and not generate_dense:
    print("[INFO] Loading all data arrays from cache...")
    for name in network_names:
        safe_name = get_safe_name(name)
        for N in N_values:
            main_data = np.loadtxt(os.path.join(data_dir, f"main_{safe_name}_N{N}.dat"))
            beta_values = main_data[:, 0]
            results_rho[name][N] = main_data[:, 1].tolist()
            results_rho_err[name][N] = main_data[:, 2].tolist()
            results_chi[name][N] = main_data[:, 3].tolist()
            results_chi_err[name][N] = main_data[:, 4].tolist()

            dense_data = np.loadtxt(os.path.join(data_dir, f"dense_{safe_name}_N{N}.dat"))
            dense_results_betas[name][N] = dense_data[:, 0]
            dense_results_rho[name][N] = dense_data[:, 1]

# --- LOAD TIME SERIES IF MAIN WAS NOT GENERATED ---
if not generate_main:
    print("[INFO] Loading time series histories...")
    for name in network_names:
        safe_name = get_safe_name(name)
        for f in os.listdir(data_dir):
            if f.startswith(f"hist_{safe_name}_b") and f.endswith(".npy"):
                b_val_str = f.replace(f"hist_{safe_name}_b", "").replace(".npy", "")
                saved_time_series[name][float(b_val_str)] = np.load(os.path.join(data_dir, f))

# =========================================================================
# 4. PLOTTING PHASE (PAPER FORMATTED)
# =========================================================================
print("\n[INFO] Generating and saving high-quality paper plots...")

# Global font update for two-column paper formatting and APS tick style
plt.rcParams.update({
    'font.size': 18,           # Tamaño base general
    'axes.labelsize': 20,      # Texto de los ejes (ej. "Tasa de infección...")
    'xtick.labelsize': 16,     # Los numeritos del eje X
    'ytick.labelsize': 16,     # Los numeritos del eje Y
    'legend.fontsize': 14,     # Texto de la caja de leyenda
    'lines.linewidth': 2.5,    # Grosor de las líneas (ajustes y series)
    'lines.markersize': 8,     # Tamaño de los puntos/triángulos/cuadrados
    # APS Style Ticks global configuration
    'xtick.direction': 'in',   # Ticks hacia adentro
    'ytick.direction': 'in',   # Ticks hacia adentro
    'xtick.top': True,         # Ticks replicados en el marco superior
    'ytick.right': True        # Ticks replicados en el marco derecho
})

for name in network_names:
    safe_name = get_safe_name(name)

    # ---------------------------------------------------------
    # Plot 1: Prevalence (Phase Transition) & Barabási Zoom
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for N in N_values:
        ax.errorbar(beta_values, results_rho[name][N], yerr=results_rho_err[name][N],
                    marker='o', capsize=3, label=rf'$N={N}$')

    ax.axvline(x=beta_c_mfa, color='black', linestyle=':', label='MFA')
    ax.axvline(x=beta_c_dbmf_dict[name][N_time_series], color='gray', linestyle='-.', label='DBMF')

    ax.set_xlabel(r'Tasa de infección, $\beta$')
    ax.set_ylabel(r'Densidad estacionaria, $\langle \rho \rangle$')
    # ax.set_title(f'Transición de fase SIS - {name}')
    ax.legend(loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(plots_dir, f"1_{safe_name}_phase_transition.png"), dpi=300, bbox_inches='tight')

    # Special handling for Barabási-Albert to show the origin collapse
    if name == "Barabási-Albert":
        # 1B: Standard with Inset Zoom
        axins = ax.inset_axes([0.55, 0.08, 0.42, 0.40])  # x0, y0, width, height
        for N in N_values:
            axins.errorbar(beta_values, results_rho[name][N], marker='o', markersize=2, linewidth=1.5, capsize=2)
        axins.axvline(x=beta_c_mfa, color='black', linestyle=':')
        axins.axvline(x=beta_c_dbmf_dict[name][N_time_series], color='gray', linestyle='-.')
        axins.set_xlim(0.00, 0.015)
        axins.set_ylim(-0.01, 0.15)
        axins.locator_params(axis='both', nbins=4)
        axins.tick_params(labelsize=12, pad=4)
        axins.grid(True, alpha=0.3)
        ax.indicate_inset_zoom(axins, edgecolor="black")
        ax.legend(loc="upper left", framealpha=0.9)
        fig.savefig(os.path.join(plots_dir, f"1_{safe_name}_phase_transition_WITH_INSET.png"),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 1C: Standalone Zoom Figure
        fig_z, ax_z = plt.subplots(figsize=(8, 5))
        for N in N_values:
            ax_z.errorbar(beta_values, results_rho[name][N], marker='o', capsize=3, label=rf'$N={N}$')
        ax_z.axvline(x=beta_c_mfa, color='black', linestyle=':', label='MFA')
        ax_z.axvline(x=beta_c_dbmf_dict[name][N_time_series], color='gray', linestyle='-.', label='DBMF')
        ax_z.set_xlim(0.00, 0.015)
        ax_z.set_ylim(-0.01, 0.25)
        ax_z.set_xlabel(r'Tasa de infección, $\beta$')
        ax_z.set_ylabel(r'Densidad estacionaria, $\langle \rho \rangle$')
        # ax_z.set_title(f'Zoom colapso en el origen - {name}')
        ax_z.legend(loc="upper left", framealpha=0.8)
        ax_z.grid(True, alpha=0.3)
        ax_z.locator_params(axis='both', nbins=5)
        ax_z.tick_params(axis='both', which='major', pad=8)
        fig_z.savefig(os.path.join(plots_dir, f"1_{safe_name}_phase_transition_ZOOM_ONLY.png"),
                      dpi=300, bbox_inches='tight')
        plt.close(fig_z)
    else:
        plt.close(fig)

    # ---------------------------------------------------------
    # Plot 2: Susceptibility (Finding beta_c)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for N in N_values:
        plt.errorbar(beta_values, results_chi[name][N], yerr=results_chi_err[name][N],
                     marker='s', capsize=3, label=rf'$N={N}$')
    plt.axvline(x=beta_c_mfa, color='black', linestyle=':', label='MFA')
    plt.axvline(x=beta_c_dbmf_dict[name][N_time_series], color='gray', linestyle='-.', label='DBMF')

    plt.xlabel(r'Tasa de infección, $\beta$')
    plt.ylabel(r'Susceptibilidad, $\chi$')
    # plt.title(f'Escalamiento de tamaño finito (Susceptibilidad) - {name}')
    plt.legend(loc="best", framealpha=0.8)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f"2_{safe_name}_susceptibility.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Plot 3: Critical Exponent (Log-Log) with Dense Grid Data
    # ---------------------------------------------------------
    fig_log, ax_log = plt.subplots(figsize=(10, 6))

    # Phase 3A: Plot only largest sizes first
    largest_Ns = N_values[-2:]
    for N in largest_Ns:
        beta_c = beta_c_dict[name][N]
        x_data = dense_results_betas[name][N] - beta_c
        y_data = dense_results_rho[name][N]

        # Ensure strict positivity for logs
        valid_idx = (y_data > 0) & (x_data < 1e-2)
        x_data = x_data[valid_idx][1:]
        y_data = y_data[valid_idx][1:]

        if len(x_data) > 1:
            line = ax_log.loglog(x_data, y_data, marker='^', linestyle='', label=rf'$N={N}$')[0]
            slope, intercept, _, _, _ = stats.linregress(np.log(x_data), np.log(y_data))
            fit_y = np.exp(intercept) * (x_data ** slope)
            ax_log.loglog(x_data, fit_y, linestyle='-', color=line.get_color(),
                          label=rf'Ajuste $\alpha={slope:.2f}$')

    ax_log.set_xlabel(r'$\beta - \beta_c$')
    ax_log.set_ylabel(r'$\langle \rho \rangle$')
    # ax_log.set_title(f'Escalamiento crítico (N grandes) - {name}')

    handles, labels = ax_log.get_legend_handles_labels()
    h_data = [h for h, l in zip(handles, labels) if 'Ajuste' not in l]
    l_data = [l for l in labels if 'Ajuste' not in l]
    h_fit = [h for h, l in zip(handles, labels) if 'Ajuste' in l]
    l_fit = [l for l in labels if 'Ajuste' in l]
    ax_log.legend(h_data + h_fit, l_data + l_fit, loc="best", framealpha=0.8)

    ax_log.grid(True, which="both", ls="--", alpha=0.3)
    fig_log.savefig(os.path.join(plots_dir, f"3_{safe_name}_critical_scaling_LARGE_N.png"),
                    dpi=300, bbox_inches='tight')

    # Phase 3B: Add the rest of the sizes to the same figure
    other_Ns = [N for N in N_values if N not in largest_Ns]
    for N in other_Ns:
        beta_c = beta_c_dict[name][N]
        x_data = dense_results_betas[name][N] - beta_c
        y_data = dense_results_rho[name][N]

        valid_idx = (y_data > 0) & (x_data < 1e-2)
        x_data = x_data[valid_idx][1:]
        y_data = y_data[valid_idx][1:]

        if len(x_data) > 1:
            line = ax_log.loglog(x_data, y_data, marker='^', linestyle='', label=rf'$N={N}$')[0]
            slope, intercept, _, _, _ = stats.linregress(np.log(x_data), np.log(y_data))
            fit_y = np.exp(intercept) * (x_data ** slope)
            ax_log.loglog(x_data, fit_y, linestyle='-', color=line.get_color(),
                          label=rf'Ajuste $\alpha={slope:.2f}$')

    # ax_log.set_title(f'Escalamiento crítico (Todos los tamaños) - {name}')

    handles, labels = ax_log.get_legend_handles_labels()
    h_data = [h for h, l in zip(handles, labels) if 'Ajuste' not in l]
    l_data = [l for l in labels if 'Ajuste' not in l]
    h_fit = [h for h, l in zip(handles, labels) if 'Ajuste' in l]
    l_fit = [l for l in labels if 'Ajuste' in l]
    ax_log.legend(h_data + h_fit, l_data + l_fit, loc="best", framealpha=0.8, fontsize=10)

    fig_log.savefig(os.path.join(plots_dir, f"3_{safe_name}_critical_scaling_ALL.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_log)

# ---------------------------------------------------------
# Plot 4: Time Evolution of Infected Fraction
# ---------------------------------------------------------
print("[INFO] Generating Plot 4: Time Evolution (Individual and Stacked)...")

# --- 4A: Individual Plots per Network ---
for name in network_names:
    safe_name = get_safe_name(name)
    cached_data = saved_time_series[name]
    if not cached_data: continue

    plt.figure(figsize=(10, 6))

    # Use 'viridis' so that low infection is dark/cold and high infection is bright/hot
    cmap = plt.get_cmap('viridis')
    sorted_betas = sorted(cached_data.keys())

    # Setup normalization and ScalarMappable for the colorbar
    norm = plt.Normalize(vmin=min(sorted_betas), vmax=max(sorted_betas))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for b in sorted_betas:
        history = cached_data[b]
        color = cmap(norm(b))
        plt.plot(range(len(history)), history, color=color, alpha=0.8)

    # Conditionally plot the transient line and its legend
    if plot_transient_line:
        plt.axvline(x=transient, color='red', linestyle='--', alpha=0.7, label='Fin del transitorio')
        plt.legend(loc="best", framealpha=0.8, fontsize=11)

    plt.xlabel(r'Tiempo, $t$')
    plt.ylabel(r'Densidad de infectados, $\rho(t)$')
    # plt.title(f'Evolución temporal ($N={N_time_series}$) - {name}')

    # Add colorbar
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(r'Tasa de infección, $\beta$')

    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f"4_{safe_name}_time_evolution.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- 4B: Stacked Multipanel Plot ---
# Check if at least one network has data to prevent empty plots
has_data = any(bool(saved_time_series[n]) for n in network_names)

if has_data:
    # Create a figure with 3 subplots sharing the X axis
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.subplots_adjust(hspace=0.08)  # Reduce vertical space between plots

    cmap = plt.get_cmap('viridis')

    # Get min and max beta across all networks for a unified colorbar
    all_betas = []
    for name in network_names:
        all_betas.extend(saved_time_series[name].keys())

    norm = plt.Normalize(vmin=min(all_betas), vmax=max(all_betas))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Panel identifiers for paper reference
    panel_letters = ['(a)', '(b)', '(c)']

    for idx, name in enumerate(network_names):
        ax = axes[idx]
        cached_data = saved_time_series[name]

        if not cached_data:
            continue

        sorted_betas = sorted(cached_data.keys())
        for b in sorted_betas:
            history = cached_data[b]
            color = cmap(norm(b))
            ax.plot(range(len(history)), history, color=color, alpha=0.8)

        # Conditionally plot the transient line and legend
        if plot_transient_line:
            ax.axvline(x=transient, color='red', linestyle='--', alpha=0.7, label='Fin del transitorio')
            # Legend only on the first panel
            if idx == 0:
                ax.legend(loc="lower right", framealpha=0.9, fontsize=11)

        ax.grid(True, alpha=0.3)
        ax.locator_params(axis='y', nbins=4)  # Prevent Y-axis ticks from overlapping

        # Professional panel identifier and network name (No bounding box, clean text)
        ax.text(0.02, 0.90, f"{panel_letters[idx]} {name}", transform=ax.transAxes, fontsize=16,
                verticalalignment='top', fontweight='bold')

    # Single global Y-axis label for the entire stacked figure
    fig.supylabel(r'Densidad de infectados, $\rho(t)$', fontsize=20)

    # X-axis label only on the bottom plot
    axes[-1].set_xlabel(r'Tiempo, $t$')

    # Add a single global colorbar for all subplots combined
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, aspect=40)
    cbar.set_label(r'Tasa de infección, $\beta$')

    # Save the stacked multipanel plot
    plt.savefig(os.path.join(plots_dir, f"4_time_evolution_STACKED_N{N_time_series}.png"), dpi=300,
                bbox_inches='tight')
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
sorted_names_for_legend = ["Erdős-Rényi", "Watts-Strogatz", "Barabási-Albert"]

# ---------------------------------------------------------
# Plot 5: Comparison Prevalence (Phase Transition)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plot_handles = {}

for name in sorted_names_for_legend:
    if name not in results_rho: continue
    net_color = comparison_color_map.get(name, "black")

    line = plt.errorbar(beta_values, results_rho[name][N_time_series],
                        yerr=results_rho_err[name][N_time_series],
                        marker='o', color=net_color, capsize=3, label=f'{name}')
    plot_handles[(name, 'data')] = line

    dbmf_line = plt.axvline(x=beta_c_dbmf_dict[name][N_time_series],
                            color=net_color, linestyle='-.', alpha=0.7, label=f'DBMF ({name})')
    plot_handles[(name, 'dbmf')] = dbmf_line

mfa_line = plt.axvline(x=beta_c_mfa, color='black', linestyle=':', label='MFA')

all_handles = [plot_handles[(n, 'data')] for n in sorted_names_for_legend if (n, 'data') in plot_handles] + \
              [plot_handles[(n, 'dbmf')] for n in sorted_names_for_legend if (n, 'dbmf') in plot_handles] + [mfa_line]

plt.xlabel(r'Tasa de infección, $\beta$')
plt.ylabel(r'Densidad estacionaria, $\langle \rho \rangle$')
# plt.title(f'Comparativa de transición de fase epidémica ($N={N_time_series}$)')
plt.legend(handles=all_handles, loc="best", framealpha=0.8)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plots_dir, f"5_comparison_phase_transition_N{N_time_series}.png"),
            dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Plot 6: Comparison Susceptibility
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plot_handles = {}

for name in sorted_names_for_legend:
    if name not in results_chi: continue
    net_color = comparison_color_map.get(name, "black")

    line = plt.errorbar(beta_values, results_chi[name][N_time_series],
                        yerr=results_chi_err[name][N_time_series],
                        marker='s', color=net_color, capsize=3, label=f'{name}')
    plot_handles[(name, 'data')] = line

    dbmf_line = plt.axvline(x=beta_c_dbmf_dict[name][N_time_series],
                            color=net_color, linestyle='-.', alpha=0.7, label=f'DBMF ({name})')
    plot_handles[(name, 'dbmf')] = dbmf_line

mfa_line = plt.axvline(x=beta_c_mfa, color='black', linestyle=':', label='MFA')

all_handles = [plot_handles[(n, 'data')] for n in sorted_names_for_legend if (n, 'data') in plot_handles] + \
              [plot_handles[(n, 'dbmf')] for n in sorted_names_for_legend if (n, 'dbmf') in plot_handles] + [mfa_line]

plt.xlabel(r'Tasa de infección, $\beta$')
plt.ylabel(r'Susceptibilidad, $\chi$')
# plt.title(f'Comparativa de picos de susceptibilidad ($N={N_time_series}$)')
plt.legend(handles=all_handles, loc="best", framealpha=0.8)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plots_dir, f"6_comparison_susceptibility_N{N_time_series}.png"),
            dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Plot 7: Comparison Finite-Size Scaling (Log-Log with Dense Grid)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

for name in sorted_names_for_legend:
    net_color = comparison_color_map.get(name, "black")
    beta_c = beta_c_dict[name][N_time_series]

    x_data = dense_results_betas[name][N_time_series] - beta_c
    y_data = dense_results_rho[name][N_time_series]

    valid_idx = (y_data > 0) & (x_data < 1e-2)
    x_data = x_data[valid_idx][1:]
    y_data = y_data[valid_idx][1:]

    if len(x_data) > 1:
        plt.loglog(x_data, y_data, marker='^', linestyle='', color=net_color, label=f'{name}')
        slope, intercept, _, _, _ = stats.linregress(np.log(x_data), np.log(y_data))
        fit_y = np.exp(intercept) * (x_data ** slope)
        plt.loglog(x_data, fit_y, linestyle='-', color=net_color, label=rf'Ajuste $\alpha={slope:.2f}$')

plt.xlabel(r'$\beta - \beta_c$')
plt.ylabel(r'$\langle \rho \rangle$')
# plt.title(f'Comparativa de escalamiento crítico ($N={N_time_series}$)')

handles, labels = plt.gca().get_legend_handles_labels()
h_data = [h for h, l in zip(handles, labels) if 'Ajuste' not in l]
l_data = [l for l in labels if 'Ajuste' not in l]
h_fit = [h for h, l in zip(handles, labels) if 'Ajuste' in l]
l_fit = [l for l in labels if 'Ajuste' in l]
plt.legend(h_data + h_fit, l_data + l_fit, loc="best", framealpha=0.8)

plt.grid(True, which="both", ls="--", alpha=0.3)
plt.savefig(os.path.join(plots_dir, f"7_comparison_critical_scaling_N{N_time_series}.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print(f"[SUCCESS] All tasks finished. Check your data at: ./{plots_dir}/")
print("=" * 60)