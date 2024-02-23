""" CS5340 Lab 3: Hidden Markov Models
See accompanying PDF for instructions.

Name: ZHANG ENHAO
Email: e1132290@u.nus.edu
Student ID: A0276557M
"""
import numpy as np
import scipy.stats
from scipy.special import softmax
from sklearn.cluster import KMeans


def initialize(n_states, x):
    """Initializes starting value for initial state distribution pi
    and state transition matrix A.

    A and pi are initialized with random starting values which satisfies the
    summation and non-negativity constraints.
    """
    seed = 5340
    np.random.seed(seed)

    pi = np.random.random(n_states)
    A = np.random.random([n_states, n_states])

    # We use softmax to satisify the summation constraints. Since the random
    # values are small and similar in magnitude, the resulting values are close
    # to a uniform distribution with small jitter
    pi = softmax(pi)
    A = softmax(A, axis=-1)

    # Gaussian Observation model parameters
    # We use k-means clustering to initalize the parameters.
    x_cat = np.concatenate(x, axis=0)
    kmeans = KMeans(n_clusters=n_states, random_state=seed).fit(x_cat[:, None])
    mu = kmeans.cluster_centers_[:, 0]
    std = np.array([np.std(x_cat[kmeans.labels_ == l]) for l in range(n_states)])
    phi = {'mu': mu, 'sigma': std}

    return pi, A, phi


"""E-step"""
def e_step(x_list, pi, A, phi):
    """ E-step: Compute posterior distribution of the latent variables,
    p(Z|X, theta_old). Specifically, we compute
      1) gamma(z_n): Marginal posterior distribution, and
      2) xi(z_n-1, z_n): Joint posterior distribution of two successive
         latent states

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        pi (np.ndarray): Current estimated Initial state distribution (K,)
        A (np.ndarray): Current estimated Transition matrix (K, K)
        phi (Dict[np.ndarray]): Current estimated gaussian parameters

    Returns:
        gamma_list (List[np.ndarray]), xi_list (List[np.ndarray])
    """
    n_states = pi.shape[0]
    gamma_list = [np.zeros([len(x), n_states]) for x in x_list]
    xi_list = [np.zeros([len(x)-1, n_states, n_states]) for x in x_list]

    """ YOUR CODE HERE
    Use the forward-backward procedure on each input sequence to populate 
    "gamma_list" and "xi_list" with the correct values.
    Be sure to use the scaling factor for numerical stability.
    """
    for seq_idx, x in enumerate(x_list):
        T = len(x)

        # Forward procedure with scaling
        alpha = np.zeros((T, n_states))
        scaling_factors = np.zeros(T)
        alpha[0] = pi * scipy.stats.norm.pdf(x[0], phi['mu'], phi['sigma'])
        scaling_factors[0] = np.sum(alpha[0])
        alpha[0] /= scaling_factors[0]

        for t in range(1, T):
            alpha[t] = np.dot(alpha[t - 1], A) * scipy.stats.norm.pdf(x[t], phi['mu'], phi['sigma'])
            scaling_factors[t] = np.sum(alpha[t])
            alpha[t] /= scaling_factors[t]

        # Backward procedure with scaling
        beta = np.zeros((T, n_states))
        beta[-1] = np.ones(n_states)
        for t in range(T - 2, -1, -1):
            beta[t] = np.dot(A, beta[t + 1] * scipy.stats.norm.pdf(x[t + 1], phi['mu'], phi['sigma']))
            beta[t] /= scaling_factors[t + 1]

        # Compute gamma and xi
        gamma = (alpha * beta)
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        gamma_list[seq_idx] = gamma

        xi = np.zeros((T - 1, n_states, n_states))
        for t in range(T - 1):
            xi[t] = (alpha[t].reshape(-1, 1) * A) * scipy.stats.norm.pdf(x[t + 1], phi['mu'], phi['sigma']) * beta[
                t + 1]
            xi[t] /= np.sum(xi[t])
        xi_list[seq_idx] = xi

    return gamma_list, xi_list


"""M-step"""
def m_step(x_list, gamma_list, xi_list):
    """M-step of Baum-Welch: Maximises the log complete-data likelihood for
    Gaussian HMMs.
    
    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        gamma_list (List[np.ndarray]): Marginal posterior distribution
        xi_list (List[np.ndarray]): Joint posterior distribution of two
          successive latent states

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.
    """

    n_states = gamma_list[0].shape[1]
    pi = np.zeros([n_states])
    A = np.zeros([n_states, n_states])
    phi = {'mu': np.zeros(n_states),
           'sigma': np.zeros(n_states)}

    """ YOUR CODE HERE
    Compute the complete-data maximum likelihood estimates for pi, A, phi.
    """
    # Estimate pi
    for seq_idx, gamma in enumerate(gamma_list):
        pi += gamma[0, :]

    pi /= len(x_list)

    # Estimate A using xi_list
    for xi in xi_list:
        A += xi.sum(axis=0)

    # Normalize A such that each row sums to 1
    A /= A.sum(axis=1, keepdims=True)

    # Estimate mu and sigma using gamma_list and x_list
    for seq_idx, (x, gamma) in enumerate(zip(x_list, gamma_list)):
        for k in range(n_states):
            phi['mu'][k] += np.sum(gamma[:, k] * x)

    # Normalize mu
    for k in range(n_states):
        phi['mu'][k] /= sum([gamma[:, k].sum() for gamma in gamma_list])

    # Estimate the sigma based on the correctly computed mu
    for seq_idx, (x, gamma) in enumerate(zip(x_list, gamma_list)):
        for k in range(n_states):
            phi['sigma'][k] += np.sum(gamma[:, k] * (x ** 2))

    # Calculate variance using E[X^2] - E[X]^2 and then take sqrt for sigma
    for k in range(n_states):
        sum_gamma = sum([gamma[:, k].sum() for gamma in gamma_list])
        phi['sigma'][k] /= sum_gamma
        phi['sigma'][k] -= phi['mu'][k] ** 2
        phi['sigma'][k] = np.sqrt(phi['sigma'][k])

    return pi, A, phi


"""Putting them together"""
def fit_hmm(x_list, n_states):
    """Fit HMM parameters to observed data using Baum-Welch algorithm

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        n_states (int): Number of latent states to use for the estimation.

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Time-independent stochastic transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.

    """

    # We randomly initialize pi and A, and use k-means to initialize phi
    # Please do NOT change the initialization function since that will affect
    # grading
    pi, A, phi = initialize(n_states, x_list)

    """ YOUR CODE HERE
     Populate the values of pi, A, phi with the correct values. 
    """

    iteration_changes = True
    previous_phi = phi.copy()

    while iteration_changes:
        # E step
        gamma_values, xi_values = e_step(x_list, pi, A, phi)
        # M step
        pi, A, phi = m_step(x_list, gamma_values, xi_values)

        # Check if the parameters have converged
        mu_difference = np.linalg.norm(previous_phi['mu'] - phi['mu'])
        sigma_difference = np.linalg.norm(previous_phi['sigma'] - phi['sigma'])

        if mu_difference < 1e-4 and sigma_difference < 1e-4:
            iteration_changes = False
        previous_phi = phi.copy()

    return pi, A, phi

