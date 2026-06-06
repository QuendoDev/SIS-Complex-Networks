import os
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed

from src.simulators.griffiths_simulator import SISGriffithsSimulator
from abc import ABC, abstractmethod


# =========================================================================
# DISORDER MODELS
# =========================================================================

class DisorderModel(ABC):
    """
    Abstract base class for quenched disorder models to generate infection rates.

    A concrete DisorderModel must implement:
      - generate(beta, num_nodes) -> np.ndarray
          returns the generated infection rates for the nodes.
      - get_sweep_range() -> np.ndarray
          returns the array of beta values to sweep.
      - get_target_beta() -> float
          returns the target beta for Finite-Size Scaling.
      - get_folder_name() -> str
          returns a formatted string with the model name and its specific parameters.
    """

    @abstractmethod
    def generate(self, beta: float, num_nodes: int) -> np.ndarray:
        pass

    @abstractmethod
    def get_sweep_range(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_target_beta(self) -> float:
        pass

    @abstractmethod
    def get_folder_name(self) -> str:
        pass


class BimodalDisorder(DisorderModel):
    """
    A class to represent a bimodal disorder model for infection rates.

    Methods
    ----------
    generate(beta, num_nodes)
        Generates bimodal infection rates with a specified fraction of type-II nodes.
    get_sweep_range()
        Returns the predefined range of beta values to sweep.
    get_target_beta()
        Returns the target beta value for FSS analysis.
    get_folder_name()
        Returns the directory name associated with this model's parameters.
    """

    def __init__(self, q=0.6, r=0.0, sweep_min=0.02, sweep_max=0.15, sweep_points=20,
                 target_beta=0.06, q_str=None, r_str=None, target_beta_str=None):
        self.q = q
        self.r = r
        self.sweep_range = np.linspace(sweep_min, sweep_max, sweep_points)
        self.target_beta = target_beta

        self.q_str = q_str if q_str else f"{self.q:.2f}"
        self.r_str = r_str if r_str else f"{self.r:.2f}"
        self.target_beta_str = target_beta_str if target_beta_str else f"{self.target_beta:.3f}"

    def generate(self, beta: float, num_nodes: int) -> np.ndarray:
        rates = [beta, beta * self.r]
        probabilities = [1.0 - self.q, self.q]
        return np.random.choice(rates, size=num_nodes, p=probabilities)

    def get_sweep_range(self) -> np.ndarray:
        return self.sweep_range

    def get_target_beta(self) -> float:
        return self.target_beta

    def get_folder_name(self) -> str:
        return f"BimodalDisorder_q_{self.q_str}_r_{self.r_str}_target_{self.target_beta_str}"


# =========================================================================
# SIMULATION CORE
# =========================================================================

def run_griffiths_decay(beta: float, matrix_adj: Any, mu: float, steps: int,
                        num_seeds: int, seed: int, N: int, disorder_model: DisorderModel) -> np.ndarray:
    """
    Execute multiple simulation runs to compute the ensemble average decay.

    :param beta: float, maximum infection rate to simulate
    :param matrix_adj: Any, adjacency matrix of the graph
    :param mu: float, recovery rate
    :param steps: int, total number of time steps to simulate
    :param num_seeds: int, number of independent simulation runs to average
    :param seed: int, random seed base
    :param N: int, number of nodes in the network
    :param disorder_model: DisorderModel, the disorder model instance
    :return: numpy array, ensemble average of the infected density over time
    """
    temp_histories = []
    for s in range(num_seeds):
        current_seed = seed + s
        np.random.seed(current_seed)
        random.seed(current_seed)

        betas = disorder_model.generate(beta, N)
        local_sim = SISGriffithsSimulator()
        rho_history = local_sim.run(matrix_adj, betas, mu, steps, initial_fraction=1.0)
        temp_histories.append(rho_history)

    return np.mean(temp_histories, axis=0)


# =========================================================================
# PARALLEL EXECUTION WRAPPERS
# =========================================================================

def simulate_and_save(b: float, matrix: Any, mu: float, steps: int,
                      num_seeds: int, seed: int, N: int, model: DisorderModel,
                      save_dir: str) -> tuple[float, np.ndarray]:
    """
    Simulate the decay for a given beta and immediately save the result to disk.

    :param b: float, maximum infection rate to simulate
    :param matrix: Any, adjacency matrix of the graph
    :param mu: float, recovery rate
    :param steps: int, total number of time steps to simulate
    :param num_seeds: int, number of independent simulation runs to average
    :param seed: int, random seed base
    :param N: int, number of nodes in the network
    :param model: DisorderModel, the disorder model instance used to generate infection rates
    :param save_dir: str, directory path where the .npy file will be safely stored
    :return: tuple, containing the evaluated b and the resulting density array
    """
    # Execute the core simulation
    decay_curve = run_griffiths_decay(b, matrix, mu, steps, num_seeds, seed, N, model)

    # Save to disk immediately to secure the data
    file_path = os.path.join(save_dir, f"sweep_b{b:.4f}.npy")
    np.save(file_path, decay_curve)

    return b, decay_curve


if __name__ == '__main__':
    print("=" * 60)
    print("       SIS Model - Griffiths Phase Exploration")
    print("=" * 60)


    # Helper functions for user input with defaults
    def prompt_float(msg: str, default: float) -> tuple[float, str]:
        """
        Prompt the user for a float value, accepting fractions.

        :param msg: str, the message to display
        :param default: float, the default value if input is empty
        :return: tuple, the parsed float value and its raw string representation
        """
        val = input(f"    {msg} (default {default}): ").strip()
        if not val:
            return default, str(default)

        if '/' in val:
            num, den = val.split('/')
            return float(num) / float(den), val
        return float(val), val


    def prompt_int(msg: str, default: int) -> int:
        """
        Prompt the user for an integer value.

        :param msg: str, the message to display
        :param default: int, the default value if input is empty
        :return: int, the parsed integer value
        """
        val = input(f"    {msg} (default {default}): ").strip()
        return int(val) if val else default

    # 1. USER INTERFACE & DISORDER SELECTION
    print("Select Disorder Model to run:")
    print("  [1] Bimodal Disorder (like Muñoz et al.)")
    print("  [!] Run ALL models sequentially")
    model_choice = input("[?] Enter your choice (#/!): ").strip()

    models_to_run = []
    print("\n[INFO] --- Setup Model Parameters ---")
    print("       (Press Enter to keep the default value)")

    if model_choice in ['1', '!']:
        print("\n  > BIMODAL DISORDER SETUP:")
        q_val, q_raw = prompt_float("Fraction q (Type-II nodes)", 0.6)
        r_val, r_raw = prompt_float("Ratio r (beta_II / beta_I)", 0.0)
        s_min, _ = prompt_float("Sweep range MIN", 0.02)
        s_max, _ = prompt_float("Sweep range MAX", 0.15)
        s_pts = prompt_int("Sweep points (grid density)", 20)
        t_beta, t_beta_raw = prompt_float("Target beta (for FSS Plot)", 0.06)

        # Replace '/' with '-' for all variables used in the directory name
        q_str = q_raw.replace('/', '-')
        r_str = r_raw.replace('/', '-')
        t_beta_str = t_beta_raw.replace('/', '-')

        models_to_run.append(BimodalDisorder(q=q_val, r=r_val, sweep_min=s_min, sweep_max=s_max, sweep_points=s_pts,
                                             target_beta=t_beta, q_str=q_str, r_str=r_str,
                                             target_beta_str=t_beta_str))

    print("\nSelect data generation mode:")
    print("  [1] Use existing data from cache (Generate NOTHING, just plot)")
    print("  [2] Generate data (Keep existing files, SKIP if already computed)")
    print("  [3] Generate data (OVERWRITE existing files)")
    print("  [4] Generate data (CLEAR ALL existing cache first)")
    data_choice = input("[?] Enter your choice (1/2/3/4): ").strip()

    generate_main = data_choice in ['2', '3', '4']
    overwrite_mode = data_choice == '3'
    clear_cache = data_choice == '4'

    do_fss_sim = True
    do_sweep_sim = True
    if generate_main:
        print("\nSelect simulation tasks to perform:")
        print("  [1] Both (FSS and Beta Sweep)")
        print("  [2] ONLY Finite-Size Scaling (FSS - different N values)")
        print("  [3] ONLY Beta Sweep (Different beta for largest N)")
        task_choice = input("[?] Enter your choice (1/2/3) [default 1]: ").strip()
        if task_choice == '2':
            do_sweep_sim = False
        elif task_choice == '3':
            do_fss_sim = False

    # 2. PARAMETERS & INITIALIZATION (Global)
    N_values = [1000, 5000, 10000, 20000, 50000, 100000]
    N_time_series = N_values[-1]
    k_avg = 3
    mu_value = 0.3
    steps = 20000  # Crucial for observing the long algebraic tail
    num_seeds = 30  # High number of seeds to smooth quenched disorder noise
    seed = 42
    n_cores = 12

    # Loop through the selected disorder models
    for active_disorder_model in models_to_run:
        # Get the specific folder name including the parameters
        folder_name = active_disorder_model.get_folder_name()

        print("\n" + "*" * 60)
        print(f"       Evaluating Disorder Model: {folder_name}")
        print("*" * 60)

        beta_values = active_disorder_model.get_sweep_range()
        target_beta = active_disorder_model.get_target_beta()

        # Directories scoped by model AND parameters to avoid overwriting cache
        data_dir = os.path.join("../../results_griffiths", "data_cache", folder_name)
        plots_dir = os.path.join("../../results_griffiths", "plots", folder_name)

        if generate_main:
            os.makedirs(data_dir, exist_ok=True)
            if clear_cache:
                for f in os.listdir(data_dir):
                    os.remove(os.path.join(data_dir, f))
                print(f"[INFO] Cleared cache. Generating new data into: {data_dir}")
            else:
                print(f"[INFO] Using existing cache directory: {data_dir}")
        else:
            if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
                print("[ERROR] No cached data found. Defaulting to Generate ALL (Skip mode).")
                generate_main = True
                overwrite_mode = False
                os.makedirs(data_dir, exist_ok=True)
            else:
                print(f"[INFO] Preparing to load data from cache: {data_dir}")

        os.makedirs(plots_dir, exist_ok=True)

        # Data structures
        results_fss = {}
        results_beta_sweep = {}

        # 3. DATA GENERATION / LOADING
        if generate_main:
            print(f"[INFO] Target beta array spans from {beta_values[0]:.3f} to {beta_values[-1]:.3f}")

            for N in N_values:
                if do_fss_sim or (do_sweep_sim and N == N_time_series):
                    print(f"\n[INFO] Evaluating system size N={N}...")

                fss_path = os.path.join(data_dir, f"fss_N{N}.npy")

                # Verify what needs to be calculated
                need_fss = do_fss_sim and (overwrite_mode or not os.path.exists(fss_path))

                betas_to_run = []
                if do_sweep_sim and (N == N_time_series):
                    for b in beta_values:
                        sweep_path = os.path.join(data_dir, f"sweep_b{b:.4f}.npy")
                        if overwrite_mode or not os.path.exists(sweep_path):
                            betas_to_run.append(b)

                # Generate network only if there's actual work to do
                if need_fss or (N == N_time_series and betas_to_run):
                    num_edges = int((N * k_avg) / 2)
                    rows = np.random.randint(0, N, size=num_edges)
                    cols = np.random.randint(0, N, size=num_edges)
                    data = np.ones(num_edges)

                    matrix_adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
                    matrix_adj = matrix_adj + matrix_adj.transpose()
                    matrix_adj.setdiag(0)
                    matrix_adj.eliminate_zeros()
                    matrix_adj.data = np.ones_like(matrix_adj.data)

                # Phase A: Finite-Size Scaling target beta
                if need_fss:
                    print(f"       --> Simulating FSS for target beta = {target_beta:.4f}...")
                    decay_curve = run_griffiths_decay(target_beta, matrix_adj, mu_value, steps,
                                                      num_seeds, seed, N, active_disorder_model)
                    results_fss[N] = decay_curve
                    np.save(fss_path, decay_curve)
                else:
                    if os.path.exists(fss_path):
                        if do_fss_sim:
                            print(f"       [SKIP] FSS data for N={N} already exists. Loading from cache...")
                        results_fss[N] = np.load(fss_path)

                # Phase B: Full sweep for the largest N
                if do_sweep_sim and (N == N_time_series):
                    # Silently load skipped betas from cache
                    skipped_count = 0
                    for b in beta_values:
                        if b not in betas_to_run:
                            sweep_path = os.path.join(data_dir, f"sweep_b{b:.4f}.npy")
                            results_beta_sweep[b] = np.load(sweep_path)
                            skipped_count += 1

                    if skipped_count > 0:
                        print(f"       [INFO] Skipped {skipped_count} existing betas from cache.")

                    if betas_to_run:
                        print(f"       --> Simulating {len(betas_to_run)} new beta values in parallel...")

                        # Execute parallel simulations using the default multiprocessing backend ('loky')
                        # to bypass the Python GIL and utilize all physical cores efficiently.
                        sweep_results_list = Parallel(n_jobs=n_cores)(
                            delayed(simulate_and_save)(
                                b, matrix_adj, mu_value, steps, num_seeds, seed, N,
                                active_disorder_model, data_dir
                            )
                            for b in betas_to_run
                        )

                        # Since the wrapper function already saved the .npy files to disk,
                        # we only need to populate the dictionary for the plotting phase.
                        for b, decay in sweep_results_list:
                            results_beta_sweep[b] = decay

        else:
            print("[INFO] Loading data from arrays...")

            for N in N_values:
                fss_path = os.path.join(data_dir, f"fss_N{N}.npy")
                if os.path.exists(fss_path):
                    results_fss[N] = np.load(fss_path)
                else:
                    print(f"       [WARNING] Missing FSS data for N={N}.")

            # Load Sweep: Find all available betas in cache
            available_betas = []
            for f in os.listdir(data_dir):
                if f.startswith("sweep_b") and f.endswith(".npy"):
                    try:
                        b_str = f.replace("sweep_b", "").replace(".npy", "")
                        available_betas.append(float(b_str))
                    except ValueError:
                        pass

            available_betas.sort()

            if not available_betas:
                print("       [ERROR] No sweep data found in cache!")
            else:
                print(f"\n[INFO] Available beta values in cache:")
                for i, b in enumerate(available_betas):
                    print(f"  [{i}] {b:.4f}")

                print("\nSelect betas to load (e.g., 'all' or '0,1,4-7,10'):")
                selection = input("[?] Selection (default 'all'): ").strip().lower()

                if selection == "" or selection == "all":
                    beta_values = available_betas
                else:
                    selected_indices = []
                    for part in selection.split(","):
                        part = part.strip()
                        if "-" in part:
                            start_idx, end_idx = part.split("-")
                            selected_indices.extend(range(int(start_idx), int(end_idx) + 1))
                        elif part.isdigit():
                            selected_indices.append(int(part))

                    # Update beta_values with only the selected ones
                    beta_values = [available_betas[i] for i in selected_indices if 0 <= i < len(available_betas)]

                print(f"       [INFO] Selected {len(beta_values)} betas for plotting.")

                for b in beta_values:
                    results_beta_sweep[b] = np.load(os.path.join(data_dir, f"sweep_b{b:.4f}.npy"))

        # =========================================================================
        # 4. PLOTTING PHASE (APS PAPER FORMATTED)
        # =========================================================================
        print("\n[INFO] Generating and saving high-quality paper plots...")

        # Global font update and APS tick configuration
        plt.rcParams.update({
            'font.size': 18,
            'axes.labelsize': 20,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 14,
            'lines.linewidth': 1,
            'lines.markersize': 4,
            # APS Style Ticks global configuration
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.top': True,
            'ytick.right': True
        })

        t_array = np.arange(1, steps + 1)

        # ---------------------------------------------------------
        # Plot 1: Finite-Size Cutoff in the Griffiths Phase
        # ---------------------------------------------------------
        fig_fss, ax_fss = plt.subplots(figsize=(10, 6))
        cmap_fss = plt.get_cmap('viridis')
        colors_fss = cmap_fss(np.linspace(0, 0.9, len(N_values)))

        for idx, n_val in enumerate(N_values):
            if n_val in results_fss:
                rho_data = results_fss[n_val]
                pos_indices = rho_data > 0
                ax_fss.loglog(t_array[pos_indices], rho_data[pos_indices],
                              color=colors_fss[idx], label=rf'$N = {n_val}$')

        ax_fss.set_xlabel(r'Tiempo, $t$')
        ax_fss.set_ylabel(r'Densidad de infectados, $\rho(t)$')

        ax_fss.legend(loc="lower left", framealpha=0.8, fontsize=12)
        ax_fss.tick_params(pad=8)  # Direction 'in' is handled globally
        ax_fss.grid(True, which="both", ls="--", alpha=0.3)

        plt.savefig(os.path.join(plots_dir, "1_griffiths_finite_size.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_fss)

        # ---------------------------------------------------------
        # Plot 2: Phase Diagram Sweep (Absorbing -> Griffiths -> Active)
        # ---------------------------------------------------------
        fig_sweep, ax_sweep = plt.subplots(figsize=(10, 6))
        cmap_sweep = plt.get_cmap('viridis')

        norm = plt.Normalize(vmin=min(beta_values), vmax=max(beta_values))
        sm = plt.cm.ScalarMappable(cmap=cmap_sweep, norm=norm)
        sm.set_array([])

        for b in beta_values:
            if b in results_beta_sweep:
                rho_history = results_beta_sweep[b]
                pos_indices = rho_history > 0
                ax_sweep.loglog(t_array[pos_indices], rho_history[pos_indices],
                                color=cmap_sweep(norm(b)), alpha=0.8)

        ax_sweep.set_xlabel(r'Tiempo, $t$')
        ax_sweep.set_ylabel(r'Densidad de infectados, $\rho(t)$')

        cbar = plt.colorbar(sm, ax=ax_sweep)
        cbar.set_label(r'Tasa de infección máxima, $\beta_{max}$')

        ax_sweep.tick_params(pad=8)
        ax_sweep.grid(True, which="both", ls="--", alpha=0.3)

        plt.savefig(os.path.join(plots_dir, f"2_griffiths_sweep_N{N_time_series}.png"), dpi=300,
                    bbox_inches='tight')
        plt.close(fig_sweep)

        # ---------------------------------------------------------
        # Plot 3: Stationary Density vs Beta (Macroscopic Transition)
        # ---------------------------------------------------------
        fig_stat, ax_stat = plt.subplots(figsize=(10, 6))

        rho_stationary = []
        plot_betas = []
        # Average the last 10% of the simulation steps to find the steady state
        steady_window = int(steps * 0.1)

        for b in beta_values:
            if b in results_beta_sweep:
                history = results_beta_sweep[b]
                steady_val = np.mean(history[-steady_window:])
                rho_stationary.append(steady_val)
                plot_betas.append(b)

        if plot_betas:
            ax_stat.plot(plot_betas, rho_stationary, marker='o', color='navy')

        ax_stat.set_xlabel(r'Tasa de infección máxima, $\beta_{max}$')
        ax_stat.set_ylabel(r'Densidad estacionaria, $\langle \rho(\infty) \rangle$')

        ax_stat.locator_params(axis='both', nbins=5)
        ax_stat.tick_params(pad=8)
        ax_stat.grid(True, ls="--", alpha=0.3)

        plt.savefig(os.path.join(plots_dir, f"3_griffiths_stationary_N{N_time_series}.png"), dpi=300,
                    bbox_inches='tight')
        plt.close(fig_stat)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] All tasks finished. Check your data at: ./results_griffiths/plots/")
    print("=" * 60)