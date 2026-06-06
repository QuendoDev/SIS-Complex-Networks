import numpy as np
from typing import Any
from numpy import ndarray


class SISGriffithsSimulator:
    """
    A class to run simulations of the SIS model with heterogeneous infection rates to explore the Griffiths phase.

    Methods
    ----------
    run(matrix_adj, betas, mu, steps, initial_fraction)
        Runs the simulation for the given network using node-specific infection rates and tracks time evolution.
    """

    def __init__(self):
        pass

    def run(self, matrix_adj: Any, betas: np.ndarray, mu: float, steps: int,
            initial_fraction: float = 1.0) -> ndarray:
        """
        Simulate the SIS model using heterogeneous epidemic parameters per node.

        :param matrix_adj: numpy array (N, N), adjacency matrix of the graph or sparse matrix
        :param betas: numpy array (N,), intrinsic infection rate for each individual node
        :param mu: float, recovery rate
        :param steps: int, total number of time steps to simulate
        :param initial_fraction: float, initial fraction of the network that is infected (default 1.0 for decay
                analysis)
        :return: numpy array (steps,), time series of infected fraction over the entire simulation
        """
        N = matrix_adj.shape[0]

        # Initialize network state based on initial_fraction
        states = np.random.choice([0, 1], size=N, p=[1.0 - initial_fraction, initial_fraction])

        # Pre-allocate array for speed (faster than list.append)
        rho_series = np.zeros(steps)

        for t in range(steps):
            rho_actual = np.sum(states) / N
            rho_series[t] = rho_actual

            # Early stopping: if the disease dies out completely
            if rho_actual == 0.0:
                break

            # Calculate the number of infected neighbors for each node
            infected_n = matrix_adj @ states

            # Element-wise calculation of infection probability.
            # Each node uses its own beta from the 'betas' array.
            p_inf = 1.0 - (1.0 - betas) ** infected_n

            # Stochastic decisions
            rand_inf = np.random.rand(N)
            rand_rec = np.random.rand(N)

            states_new = states.copy()

            # Infection process
            infecting = (states == 0) & (rand_inf < p_inf)
            states_new[infecting] = 1

            # Recovering process
            recovering = (states == 1) & (rand_rec < mu)
            states_new[recovering] = 0

            states = states_new

        return rho_series