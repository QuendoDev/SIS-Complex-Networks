import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import List, Tuple


def get_theoretical_regime(q: float, beta: float, mu: float = 0.3, k_avg: float = 3.0) -> str:
    """
    Determine the theoretical physical regime for a given beta and q.

    :param q: float, fraction of type-II (inactive) nodes
    :param beta: float, infection rate
    :param mu: float, recovery rate
    :param k_avg: float, average degree of the network
    :return: str, the name of the theoretical regime
    """
    q_perc = 1.0 - (1.0 / k_avg)
    beta_c_0 = mu * (1.0 / (k_avg + 1.0))
    beta_c_q_perc = beta_c_0 / (1.0 - q_perc)

    if beta < beta_c_0:
        return "Pure Exponential"

    if q < q_perc:
        # Connected network (Percolating)
        beta_c_q = beta_c_0 / (1.0 - q)
        if beta < beta_c_q:
            return "Stretched Exponential"
        else:
            return "Active Phase (Endemic)"
    elif q > q_perc:
        # Fragmented network (Non-Percolating)
        if beta < beta_c_q_perc:
            return "Stretched Exponential"
        else:
            return "Griffiths Phase"
    else:
        # Marginal case (Exactly at q_perc)
        if beta < beta_c_q_perc:
            return "Stretched Exponential"
        else:
            return "Logarithmic Decay (Marginal)"


def gather_available_data(base_dir: str, q_str: str) -> List[Tuple[float, str]]:
    """
    Scan the cache directories to find all simulated beta values for a specific q.

    :param base_dir: str, root directory where data cache is stored
    :param q_str: str, string representation of q used in folder names (e.g., '0.30' or '2-3')
    :return: list of tuples, containing (beta, file_path) sorted by beta
    """
    available_files = []

    if not os.path.exists(base_dir):
        return available_files

    # Search through all folders that match the target q
    search_prefix = f"BimodalDisorder_q_{q_str}_r_0.00"

    for folder in os.listdir(base_dir):
        if folder.startswith(search_prefix):
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path):
                # Search for sweep files
                for file in os.listdir(folder_path):
                    if file.startswith("sweep_b") and file.endswith(".npy"):
                        try:
                            b_str = file.replace("sweep_b", "").replace(".npy", "")
                            beta_val = float(b_str)
                            file_path = os.path.join(folder_path, file)
                            available_files.append((beta_val, file_path))
                        except ValueError:
                            pass

    # Remove duplicates (in case multiple folders have the same beta sweep)
    unique_files = {}
    for beta_val, file_path in available_files:
        unique_files[beta_val] = file_path

    sorted_files = sorted(unique_files.items())
    return sorted_files


def plot_summary_spectrum(q_val: float, q_str: str, selected_data: List[Tuple[float, str]],
                          output_dir: str, steps: int = 20000, suffix: str = "",
                          show_thresholds: bool = False, full_data: List[Tuple[float, str]] = None,
                          mu: float = 0.3, k_avg: float = 3.0) -> None:
    """
    Generate and save a summary log-log plot with a color gradient and theoretical lines.

    :param q_val: float, numerical value of q
    :param q_str: str, string representation of q for saving files
    :param selected_data: list of tuples, selected (beta, file_path) to plot
    :param output_dir: str, directory path to save the generated plot
    :param steps: int, total number of simulation steps
    :param suffix: str, suffix to append to the output filename
    :param show_thresholds: bool, if True, theoretical boundary lines will be plotted
    :param full_data: list of tuples, complete dataset to extract the closest threshold curves
    :param mu: float, recovery rate used for threshold calculations
    :param k_avg: float, average degree used for threshold calculations
    :return: None
    """
    if not selected_data:
        print(f"[WARNING] No data provided to plot for {suffix}.")
        return

    # APS Style Configuration
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'lines.linewidth': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True
    })

    fig, ax = plt.subplots(figsize=(10, 7))

    # Extract betas for colormap mapping
    betas = [b for b, _ in selected_data]
    min_b, max_b = min(betas), max(betas)

    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=min_b, vmax=max_b)

    t_array = np.arange(1, steps + 1)

    # Plot empirical data
    for beta_val, file_path in selected_data:
        rho_data = np.load(file_path)
        pos_indices = rho_data > 0

        color = cmap(norm(beta_val))
        ax.loglog(t_array[pos_indices], rho_data[pos_indices], color=color, alpha=0.85)

    # Add theoretical threshold boundary curves if requested
    # --------------------------------------------------
    if show_thresholds and full_data:
        q_perc = 1.0 - (1.0 / k_avg)
        beta_c_0 = mu * (1.0 / (k_avg + 1.0))
        beta_c_q_perc = beta_c_0 / (1.0 - q_perc)

        thresholds_to_plot = []

        if q_val < q_perc:
            beta_c_q = beta_c_0 / (1.0 - q_val)
            thresholds_to_plot.append((rf"Suelo Crítico ($\beta_c(0) = {beta_c_0:.3f}$)", beta_c_0, 'darkred'))
            thresholds_to_plot.append((rf"Línea Crítica ($\beta_c(q) = {beta_c_q:.3f}$)", beta_c_q, 'black'))
        elif q_val > q_perc:
            thresholds_to_plot.append((rf"Suelo Crítico ($\beta_c(0) = {beta_c_0:.3f}$)", beta_c_0, 'darkred'))
            thresholds_to_plot.append(
                (rf"Límite Fragmentado ($\beta_c(q_{{perc}}) = {beta_c_q_perc:.3f}$)", beta_c_q_perc, 'black'))

        # Search for the closest empirical curve in the full dataset for each theoretical threshold
        for label, target_b, line_color in thresholds_to_plot:
            closest_beta, file_path = min(full_data, key=lambda item: abs(item[0] - target_b))
            rho_data = np.load(file_path)
            pos_indices = rho_data > 0

            # Plot the separating line thicker, dashed, and on top of other curves
            ax.loglog(t_array[pos_indices], rho_data[pos_indices], color=line_color,
                      linestyle='--', linewidth=3.0, alpha=0.9, zorder=10,
                      label=f"{label}\n[Simulada: $\\beta={closest_beta:.3f}$]")

    # Add theoretical guiding lines (Mean-Field or Marginal) based on the q value
    # --------------------------------------------------
    if q_val < 0.66:
        t_theory = np.geomspace(5, 200, 50)
        y_theory = 1.55 * (t_theory ** -1.0)
        ax.loglog(t_theory, y_theory, color='dimgray', linestyle='--', linewidth=2.5, zorder=5,
                  label=r'$\rho \sim t^{-1}$')

    elif abs(q_val - (2 / 3)) < 0.01:
        t_theory = np.geomspace(50, 4000, 200)
        y_theory = 0.25 * (np.log(t_theory) ** -0.5)
        ax.loglog(t_theory, y_theory, color='dimgray', linestyle='--', linewidth=2.5, zorder=5,
                  label=r'$\rho \sim \ln(t)^{-1/2}$')

    # Labels and formatting
    ax.set_xlabel(r'Tiempo, $t$')
    ax.set_ylabel(r'Densidad de infectados, $\rho(t)$')

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r'Tasa de infección, $\beta$')

    # Check if there are labeled theoretical lines before drawing the legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", framealpha=0.95, edgecolor='gray', fontsize=18)

    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_ylim(bottom=1e-6, top=1.5)

    # Save Plot
    os.makedirs(output_dir, exist_ok=True)
    save_path_img = os.path.join(output_dir, f"summary_spectrum_q_{q_str}{suffix}.png")
    fig.savefig(save_path_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] Plot saved to: {os.path.basename(save_path_img)}")


if __name__ == '__main__':
    print("=" * 60)
    print("       SIS Model - Results Plotting Interface")
    print("=" * 60)

    k = 3.0
    mu = 0.3

    BASE_CACHE_DIR = os.path.join("../../results_griffiths", "data_cache")
    OUTPUT_PLOTS_DIR = os.path.join("../../results_griffiths", "plots", "summary_spectrums")

    q_input = input("[?] Enter the q value you want to plot (e.g., 0.3, 0.9, 2/3): ").strip()

    # Parse q value
    if '/' in q_input:
        num, den = q_input.split('/')
        q_val = float(num) / float(den)
        q_str = q_input.replace('/', '-')
    else:
        q_val = float(q_input)
        q_str = f"{q_val:.2f}"

    print(f"\n[INFO] Scanning cache for q = {q_val:.4f}...")
    available_data = gather_available_data(BASE_CACHE_DIR, q_str)

    if not available_data:
        print(f"[ERROR] No cached data found for q = {q_str}. (TODO: Run simulations for this q)")
        exit()

    print(f"\nAvailable beta values for q = {q_str}:")
    print("-" * 50)
    for idx, (beta_val, _) in enumerate(available_data):
        regime = get_theoretical_regime(q=q_val, beta=beta_val, mu=mu, k_avg=k)
        print(f"  [{idx:2d}] beta = {beta_val:.4f}  -->  {regime}")

    print("-" * 50)
    print("Enter the indices you want to plot, separated by commas or dashes.")
    print("Example: '0, 5, 10-15, 20'")
    # Example q=0.9: 0, 20, 39, 50, 65, 79, 80, 85, 92, 100, 110, 119
    # Example q=0.3: 0, 20, 39, 53, 68, 76, 78, 80, 84, 91, 101, 118
    # Example q=2/3: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39
    selection = input("[?] Selection: ").strip().lower()

    selected_indices = []
    if selection == "all" or selection == "":
        selected_indices = list(range(len(available_data)))
    else:
        for part in selection.split(","):
            part = part.strip()
            if "-" in part:
                start_idx, end_idx = part.split("-")
                selected_indices.extend(range(int(start_idx), int(end_idx) + 1))
            elif part.isdigit():
                selected_indices.append(int(part))

    # Filter only valid indices
    selected_data = [available_data[i] for i in selected_indices if 0 <= i < len(available_data)]

    # Check if we are outside the marginal percolation point to enable threshold plotting
    q_perc_theoretical = 1.0 - (1.0 / k)
    is_marginal = abs(q_val - q_perc_theoretical) < 0.01

    generate_all = 'n'
    if not is_marginal:
        generate_all = (input(
            "\n[?] Do you want to generate an extra plot with ALL available betas and theoretical thresholds? (y/n): ")
                        .strip().lower())

    print(f"\n[INFO] Generating requested plots...")

    # 1. Standard Plot (Selected betas, no threshold lines)
    plot_summary_spectrum(q_val, q_str, selected_data, OUTPUT_PLOTS_DIR,
                          suffix="_selected", show_thresholds=False, mu=mu, k_avg=k)

    # 2. Selected Plot WITH threshold lines (Omitted for marginal case)
    if not is_marginal:
        plot_summary_spectrum(q_val, q_str, selected_data, OUTPUT_PLOTS_DIR,
                              suffix="_selected_thresholds", show_thresholds=True, full_data=available_data,
                              mu=mu, k_avg=k)

    # 3. ALL Betas Plot WITH threshold lines
    if generate_all == 'y' and not is_marginal:
        plot_summary_spectrum(q_val, q_str, available_data, OUTPUT_PLOTS_DIR,
                              suffix="_all_thresholds", show_thresholds=True, full_data=available_data,
                              mu=mu, k_avg=k)

    # Save Beta Selection Log just once
    save_path_txt = os.path.join(OUTPUT_PLOTS_DIR, f"summary_spectrum_q_{q_str}_betas.txt")
    with open(save_path_txt, "w", encoding="utf-8") as f:
        f.write(f"Selected beta values for q = {q_str}\n")
        f.write("-" * 60 + "\n")
        for beta_val, _ in selected_data:
            regime = get_theoretical_regime(q=q_val, beta=beta_val, mu=mu, k_avg=k)
            f.write(f"beta = {beta_val:.4f}  -->  {regime}\n")

    print(f"\n[SUCCESS] Beta values log saved to: {os.path.basename(save_path_txt)}")
    print("=" * 60)