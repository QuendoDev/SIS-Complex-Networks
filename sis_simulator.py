from typing import Any

import numpy as np
import networkx as nx
from numpy import dtype, ndarray, floating


class SISSimulator:
    """
    A class to run discrete-time simulations of the SIS model on complex networks.

    Methods
    ----------
    run(matrix_adj, beta, mu, steps, transient, initial_fraction)
        Simulate the SIS model using the specified epidemic parameters and return the time series of infected fraction
        in steady state.
    """

    def __init__(self):
        pass

    def run(self, matrix_adj: Any, beta: float, mu: float, steps: int, transient: int,
            initial_fraction: float = 0.1) -> ndarray:
        """
        Simulate the SIS model using the specified epidemic parameters.

        :param matrix_adj: numpy array (N, N), adjacency matrix of the graph
        :param beta: float, epidemic parameter
        :param mu: float, epidemic parameter
        :param steps: int, number of steps
        :param transient: int, number of transient nodes
        :param initial_fraction: float, initial fraction of the network that is infected (default 0.1)
        :return: numpy array (steps,), time series of infected fraction in steady state
        """
        N = matrix_adj.shape[0]

        # Initialize randomly: S=1 - initial_fraction, I=initial_fraction
        states = np.random.choice([0, 1], size=N, p=[1.0 - initial_fraction, initial_fraction])

        rho_series = []

        for t in range(steps):
            # Save the current density of infected nodes BEFORE updating (records t=0)
            rho_actual = np.sum(states) / N
            rho_series.append(rho_actual)

            # Early stopping: if the disease dies out, fill the rest with 0s and stop
            if rho_actual == 0.0:
                missing_steps = steps - t - 1
                rho_series.extend([0.0] * missing_steps)
                break

            # Calculate the infected neighbors
            infected_n = matrix_adj @ states

            # Calculate the infection probability for each node
            p_inf = 1.0 - (1.0 - beta) ** infected_n

            # Get two random arrays for infection and recovery decisions
            rand_inf = np.random.rand(N)
            rand_rec = np.random.rand(N)

            # Copy of the states (for the next step)
            states_new = states.copy()

            # Infection process: only for susceptible nodes (state=0)
            infecting = (states == 0) & (rand_inf < p_inf)
            states_new[infecting] = 1

            # Recovering process: only for infected nodes (state=1)
            recovering = (states == 1) & (rand_rec < mu)
            states_new[recovering] = 0

            # Update states for the next iteration
            states = states_new

        return np.array(rho_series)


def compute_susceptibility(rho_series: np.ndarray, num_nodes: int) -> float | floating[Any]:
    """
    Compute the susceptibility of the system based on the variance of the infected fraction.

    :param rho_series: numpy array (T,), time series of infected fraction in steady state
    :param num_nodes: int, total number of nodes in the network N
    :return: float, computed susceptibility chi
    """
    if len(rho_series) == 0 or np.all(rho_series == 0):
        return 0.0

    rho_mean = np.mean(rho_series)
    rho_sq_mean = np.mean(rho_series ** 2)

    return num_nodes * (rho_sq_mean - rho_mean ** 2)