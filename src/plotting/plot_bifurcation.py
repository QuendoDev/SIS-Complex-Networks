import os
import numpy as np
import matplotlib.pyplot as plt


def plot_transcritical_bifurcation(mu: float, output_dir: str) -> None:
    """
    Generate and save a transcritical bifurcation diagram for the mean-field SIS model.

    :param mu: float, constant recovery rate
    :param output_dir: str, directory path to save the generated plot
    :return: None
    """
    # APS Style Configuration
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'lines.linewidth': 3.5,
        'lines.markersize': 8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True
    })

    beta_values = np.linspace(0, 1, 500)

    rho_sano_estable, rho_sano_inestable = [], []
    beta_sano_estable, beta_sano_inestable = [], []

    rho_endem_estable, rho_endem_inestable = [], []
    beta_endem_estable, beta_endem_inestable = [], []

    for beta in beta_values:
        # Branch 1: rho = 0
        if beta < mu:
            beta_sano_estable.append(beta)
            rho_sano_estable.append(0)
        else:
            beta_sano_inestable.append(beta)
            rho_sano_inestable.append(0)

        # Branch 2: rho = 1 - mu/beta
        if beta == 0:
            continue

        rho_endem = 1 - (mu / beta)

        if beta < mu:
            beta_endem_inestable.append(beta)
            rho_endem_inestable.append(rho_endem)
        else:
            beta_endem_estable.append(beta)
            rho_endem_estable.append(rho_endem)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot branches
    ax.plot(beta_sano_estable, rho_sano_estable, color='#011f4b', label='Estado Sano (Estable)')
    ax.plot(beta_endem_inestable, rho_endem_inestable, color='#6497b1', linestyle='--', alpha=0.5,
            label='Estado Endémico (Inestable)')

    ax.plot(beta_sano_inestable, rho_sano_inestable, color='#011f4b', linestyle='--', alpha=0.5,
            label='Estado Sano (Inestable)')
    ax.plot(beta_endem_estable, rho_endem_estable, color='#6497b1', label='Estado Endémico (Estable)')

    # Critical bifurcation point
    ax.scatter([mu], [0], color='red', s=150, zorder=5, label=r'Punto de Bifurcación ($\beta_c = \mu$)')
    ax.axvline(x=mu, color='gray', linestyle=':', linewidth=2)

    # Formatting
    ax.set_xlabel(r'Tasa de infección, $\beta$')
    ax.set_ylabel(r'Densidad estacionaria, $\rho^*$')

    ax.set_ylim(-0.5, 1.0)
    ax.set_xlim(0, 1.0)
    ax.axhline(y=0, color='black', linewidth=1, zorder=0)

    ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.55), framealpha=0.8)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Save Plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "Bifurcacion_SIS.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  [+] Plot saved to: {os.path.basename(save_path)}")


if __name__ == '__main__':
    print("=" * 60)
    print("       Generating Mean-Field Bifurcation Diagram")
    print("=" * 60)

    MU_VALUE = 0.2
    OUTPUT_DIRECTORY = os.path.join("../../results_griffiths", "plots")

    plot_transcritical_bifurcation(mu=MU_VALUE, output_dir=OUTPUT_DIRECTORY)

    print("=" * 60)