import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple


def gather_fss_folders(base_dir: str, q_str: str) -> List[Tuple[float, str]]:
    """
    Scan the cache directories to find all available FSS target betas for a specific q.

    :param base_dir: str, root directory where data cache is stored
    :param q_str: str, string representation of q used in folder names
    :return: list of tuples, containing (target_beta, folder_path) sorted by beta
    """
    available_folders = []

    if not os.path.exists(base_dir):
        return available_folders

    search_prefix = f"BimodalDisorder_q_{q_str}_r_0.00_target_"

    for folder in os.listdir(base_dir):
        if folder.startswith(search_prefix):
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path):
                # Extract the target beta from the folder name
                try:
                    b_str = folder.replace(search_prefix, "")
                    target_beta = float(b_str.replace("-", "."))

                    # Verify that FSS files actually exist in this folder
                    has_fss = any(f.startswith("fss_N") and f.endswith(".npy") for f in os.listdir(folder_path))
                    if has_fss:
                        available_folders.append((target_beta, folder_path))
                except ValueError:
                    pass

    return sorted(available_folders)


def gather_fss_files(folder_path: str) -> List[Tuple[int, str]]:
    """
    Scan a specific folder to find all available FSS network sizes (N).

    :param folder_path: str, path to the directory containing the simulation cache
    :return: list of tuples, containing (N_value, file_path) sorted by N
    """
    available_files = []

    for file in os.listdir(folder_path):
        if file.startswith("fss_N") and file.endswith(".npy"):
            try:
                n_str = file.replace("fss_N", "").replace(".npy", "")
                n_val = int(n_str)
                file_path = os.path.join(folder_path, file)
                available_files.append((n_val, file_path))
            except ValueError:
                pass

    return sorted(available_files)


def plot_finite_size_cutoff(fss_data: List[Tuple[int, str]], output_dir: str,
                            q_str: str, target_beta: float, steps: int = 20000) -> None:
    """
    Generate and save a finite-size scaling log-log plot to show the cutoff effect.

    :param fss_data: list of tuples, selected (N_value, file_path) sorted by N
    :param output_dir: str, directory path to save the generated plot
    :param q_str: str, string representation of q for saving files
    :param target_beta: float, the maximum infection rate used for these FSS simulations
    :param steps: int, total number of simulation steps
    :return: None
    """
    if not fss_data:
        print("[WARNING] No data provided to plot.")
        return

    # APS Style Configuration (Matching plot_summary.py)
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'lines.linewidth': 1.5,  # Thinner line to avoid thick overlapping blocks
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Apply colormap mapping smoothly across the available N values
    cmap_fss = cm.viridis
    # Using 0.0 to 0.9 prevents the highest N from being too light/yellow to see clearly
    colors_fss = cmap_fss(np.linspace(0.0, 0.9, len(fss_data)))

    t_array = np.arange(1, steps + 1)

    for idx, (n_val, file_path) in enumerate(fss_data):
        rho_data = np.load(file_path)
        pos_indices = rho_data > 0

        # Plot empirical FSS data with alpha to let the noise breathe
        ax.loglog(t_array[pos_indices], rho_data[pos_indices],
                  color=colors_fss[idx], alpha=0.85, label=rf'$N = {n_val}$')

    # Labels and formatting
    ax.set_xlabel(r'Tiempo, $t$')
    ax.set_ylabel(r'Densidad de infectados, $\rho(t)$')

    # Legend configuration
    ax.legend(loc="lower left", framealpha=0.95, edgecolor='gray')
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # Save Plot
    os.makedirs(output_dir, exist_ok=True)
    target_beta_str = f"{target_beta:.3f}".replace(".", "-")
    save_path_img = os.path.join(output_dir, f"1_griffiths_finite_size_q_{q_str}_b_{target_beta_str}.png")
    fig.savefig(save_path_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[SUCCESS] Plot saved to: {os.path.basename(save_path_img)}")


if __name__ == '__main__':
    print("=" * 60)
    print("       SIS Model - Finite-Size Cutoff Plotter")
    print("=" * 60)

    BASE_CACHE_DIR = os.path.join("../../results_griffiths", "data_cache")
    OUTPUT_PLOTS_DIR = os.path.join("../../results_griffiths", "plots", "finite_size_effects")

    q_input = input("[?] Enter the q value you want to plot FSS for (e.g., 0.9): ").strip()

    if '/' in q_input:
        num, den = q_input.split('/')
        q_str = q_input.replace('/', '-')
    else:
        q_str = f"{float(q_input):.2f}"

    print(f"\n[INFO] Scanning cache for FSS data at q = {q_str}...")
    available_folders = gather_fss_folders(BASE_CACHE_DIR, q_str)

    if not available_folders:
        print(f"[ERROR] No FSS cached data found for q = {q_str}.")
        exit()

    print(f"\nAvailable Target Betas for FSS (q = {q_str}):")
    print("-" * 50)
    for idx, (t_beta, _) in enumerate(available_folders):
        print(f"  [{idx:2d}] Target Beta = {t_beta:.3f}")

    print("-" * 50)
    selection_idx = input("[?] Select the index of the Target Beta to plot: ").strip()

    try:
        selection_idx = int(selection_idx)
        selected_beta, selected_folder = available_folders[selection_idx]
    except (ValueError, IndexError):
        print("[ERROR] Invalid selection.")
        exit()

    print(f"\n[INFO] Loading network sizes (N) from folder...")
    fss_data = gather_fss_files(selected_folder)

    if not fss_data:
        print("[ERROR] No valid fss_N*.npy files found in the selected folder.")
        exit()

    print(f"[INFO] Found {len(fss_data)} sizes: {[n for n, _ in fss_data]}")
    plot_finite_size_cutoff(fss_data, OUTPUT_PLOTS_DIR, q_str, selected_beta)