import os
import numpy as np
import matplotlib.pyplot as plt


def plot_theoretical_phase_diagram(k_avg: float, mu: float, output_dir: str, mode: str = 'full') -> None:
    """
    Generate and save a clean, explanatory theoretical phase diagram for the SIS model.
    The output details depend on the selected mode.

    Modes:
      - 'colors_only': Only zones and lines (no text, no legends, no parameter box).
      - 'full': Everything included (current state).
      - 'legend_text_only': Text in legend only, no parameter box.
      - 'legend_formulas_only': Formulas in legend only, no parameter box.
      - 'no_legend_no_params': Text inside the plot, but no legend and no parameter box.

    :param k_avg: float, average degree of the Erdős-Rényi network
    :param mu: float, recovery rate
    :param output_dir: str, directory path to save the generated plot
    :param mode: str, detail level of the generated plot
    :return: None
    """
    # APS Style Configuration
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True
    })

    # 1. Calculate q_perc and 1 - q_perc
    q_perc = 1.0 - (1.0 / k_avg)
    p_perc = 1.0 - q_perc

    # 2. Calculate beta_c at 0 and at q_perc
    dbmf_factor = mu * (1.0 / (k_avg + 1.0))
    beta_c_0 = dbmf_factor * 1.0
    beta_c_q_perc = dbmf_factor * (1.0 / p_perc)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Define array for the active region (from percolation threshold to complete network)
    p_active = np.linspace(p_perc, 1.0, 300)
    beta_c_active = dbmf_factor * (1.0 / p_active)

    # =========================================================================
    # ZONE FILLS
    # =========================================================================

    # Active Phase
    ax.fill_between(p_active, beta_c_active, 1.05, color='paleturquoise', alpha=0.8)

    # Griffiths Phase
    p_griffiths = np.linspace(0.0, p_perc, 100)
    beta_griffiths_lower = np.full_like(p_griffiths, beta_c_q_perc)
    ax.fill_between(p_griffiths, beta_griffiths_lower, 1.05, color='khaki', alpha=0.8)

    # Stretched Exponential Phase
    ax.fill_between(p_active, beta_c_0, beta_c_active, color='lightsalmon', alpha=0.8)
    ax.fill_between(p_griffiths, beta_c_0, beta_griffiths_lower, color='lightsalmon', alpha=0.8)

    # Pure Exponential Phase
    ax.fill_between(np.linspace(0, 1, 100), 0, beta_c_0, color='plum', alpha=0.8)

    # =========================================================================
    # LABEL FORMATTING BASED ON MODE
    # =========================================================================

    if mode == 'legend_text_only':
        lbl_critical = 'Línea Crítica'
        lbl_perc = 'Umbral de percolación'
        lbl_frag = 'Límite fragmentado'
        lbl_floor = 'Suelo crítico puro'
    elif mode == 'legend_formulas_only':
        lbl_critical = rf'$\beta_c(q) = \frac{{{dbmf_factor:.3f}}}{{1-q}}$'
        lbl_perc = r'$1-q_{perc} = 1/3$'
        lbl_frag = rf'$\beta_c(q_{{perc}}) = {beta_c_q_perc:.3f}$'
        lbl_floor = rf'$\beta_c(0) = {beta_c_0:.3f}$'
    else:  # 'full', 'colors_only', 'no_legend_no_params'
        lbl_critical = rf'Línea Crítica, $\beta_c(q) = \mu \frac{{\langle k \rangle}}{{\langle k^2'\
                       rf'\rangle}}\frac{{1}}{{1-q}} = \frac{{{dbmf_factor:.3f}}}{{1-q}}$'
        lbl_perc = r'Umbral de percolación, $1-q_{perc} = 1/3$'
        lbl_frag = rf'Límite fragmentado, $\beta_c(q_{{perc}}) = {beta_c_q_perc:.3f}$'
        lbl_floor = rf'Suelo crítico puro, $\beta_c(0) = {beta_c_0:.3f}$'

    # =========================================================================
    # BOUNDARIES (LINES)
    # =========================================================================

    ax.plot(p_active, beta_c_active, color='red', linewidth=3, label=lbl_critical)
    ax.plot([p_perc, p_perc], [beta_c_q_perc, 1.05], color='black', linestyle='-', linewidth=3, label=lbl_perc)

    # Dashed line for the subcritical percolation boundary (no label to avoid legend duplication)
    ax.plot([p_perc, p_perc], [0, beta_c_q_perc], color='black', linestyle='--', linewidth=2)

    ax.plot([0, p_perc], [beta_c_q_perc, beta_c_q_perc], color='blue', linestyle='-', linewidth=3, label=lbl_frag)
    ax.axhline(y=beta_c_0, color='green', linestyle=':', linewidth=3, label=lbl_floor)

    # =========================================================================
    # ANNOTATIONS & TEXT (Only if mode is NOT 'colors_only')
    # =========================================================================

    bbox_props = None

    if mode != 'colors_only':
        # Regimes (Top Headers)
        ax.text(p_perc / 2, 1.01, 'RÉGIMEN NO PERCOLANTE\n(Red fragmentada)',
                fontsize=13, ha='center', va='bottom', weight='bold')
        ax.text(p_perc + (1 - p_perc) / 2, 1.01, 'RÉGIMEN PERCOLANTE\n(Componente gigante)',
                fontsize=13, ha='center', va='bottom', weight='bold')

        # Main Phase Labels
        center_left = p_perc / 2
        center_right = p_perc + (1 - p_perc) / 2

        ax.text(center_right, 0.75, 'FASE ACTIVA\n(Estado endémico)',
                fontsize=14, ha='center', va='center', weight='bold', bbox=bbox_props)

        ax.text(center_left, 0.65, 'FASE DE GRIFFITHS\n(Decaimiento algebraico)',
                fontsize=14, ha='center', va='center', weight='bold', rotation=50, bbox=bbox_props)

        ax.text(center_left, (beta_c_q_perc + beta_c_0) / 2, 'FASE ABSORBENTE\n(Exponencial estirada)',
                fontsize=12, ha='center', va='center', weight='bold', bbox=bbox_props)

        ax.text(center_right, beta_c_0 / 2, 'FASE ABSORBENTE (Exponencial pura)',
                fontsize=12, ha='center', va='center', weight='bold', bbox=bbox_props)

        # Multicritical Point Marker
        ax.plot(p_perc, beta_c_q_perc, marker='o', color='black', markersize=10, zorder=5)
        ax.annotate('Punto\nMulticrítico', xy=(p_perc, beta_c_q_perc), xytext=(p_perc + 0.05, beta_c_q_perc + 0.08),
                    arrowprops=dict(facecolor='black', arrowstyle='-|>', lw=2),
                    fontsize=12, weight='bold', ha='left', va='center', bbox=bbox_props)

    # =========================================================================
    # LIMITS, LABELS, PARAMETERS AND GRID
    # =========================================================================

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'Fracción de nodos activos, $1 - q$')
    ax.set_ylabel(r'Tasa de infección, $\beta$')

    # Merge the two zeros at the origin by removing the 0.0 from the Y-axis ticks
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    # Display network parameters in the top-right corner only in 'full' mode
    if mode == 'full':
        ax.text(0.98, 0.97,
                r'$\langle k \rangle = 3 \quad | \quad \mu = 0.3 \quad | \quad q_{perc} = 1'
                r' - \frac{1}{\langle k \rangle} = 2/3$',
                fontsize=13, ha='right', va='top', bbox=bbox_props)

    # Display legend for applicable modes
    if mode in ['full', 'legend_text_only', 'legend_formulas_only']:
        ax.legend(loc='center right', bbox_to_anchor=(0.98, 0.5), framealpha=0.8, edgecolor='gray')

    ax.grid(True, linestyle=':', alpha=0.6)

    # Save the generated plot with a dynamic filename based on the mode
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"theoretical_phase_diagram_{mode}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  [+] Mode '{mode}' saved as: {os.path.basename(save_path)}")


if __name__ == '__main__':
    print("=" * 60)
    print("       Generating Phase Diagrams in Multiple Modes")
    print("=" * 60)

    K_AVG = 3.0
    MU_VALUE = 0.3
    OUTPUT_DIRECTORY = os.path.join("../../results_griffiths", "plots")

    # Define the 5 modes required
    modes_to_generate = [
        'colors_only',
        'full',
        'legend_text_only',
        'legend_formulas_only',
        'no_legend_no_params'
    ]

    for mode in modes_to_generate:
        plot_theoretical_phase_diagram(k_avg=K_AVG, mu=MU_VALUE, output_dir=OUTPUT_DIRECTORY, mode=mode)

    print("=" * 60)
    print("[SUCCESS] All 5 diagrams successfully generated.")
    print("=" * 60)