import numpy as np
from scipy.stats import norm, multivariate_normal
from copulas.univariate import GaussianKDE
from copulas.bivariate import Clayton, Gumbel
import matplotlib.pyplot as plt
from scipy.stats import uniform
from numpy.random import default_rng

def gaussian_copula_sample(correlation_matrix, num_samples):
    """
    Generate samples from a Gaussian copula.

    Args:
        correlation_matrix (ndarray): Correlation matrix of the copula.
        num_samples (int): Number of samples to generate.

    Returns:
        ndarray: Uniform samples in [0, 1] from the copula.
    """
    
    mean = np.zeros(correlation_matrix.shape[0])
    mvn_samples = multivariate_normal(mean, correlation_matrix).rvs(size=num_samples)
    
    uniform_samples = norm.cdf(mvn_samples)
    return uniform_samples


def archimedean_copula_sample(copula_class, num_samples):
    """
    Generate samples from an Archimedean copula.

    Args:
        copula_class: A copula class from the copulas library (e.g., Clayton, Gumbel).
        num_samples (int): Number of samples to generate.

    Returns:
        ndarray: Samples from the copula.
    """
   
    copula = copula_class()
    
    # Générer des données uniformes corrélées avec une dépendance positive
    x = np.random.uniform(0, 1, 100)
    y = 0.7 * x + 0.3 * np.random.uniform(0, 1, 100)  # Introduire une corrélation positive
    data = np.column_stack((x, y))
    copula.fit(data)
    
    samples = copula.sample(num_samples)
    return samples


def clayton_copula_multivariate(theta, num_samples, portfolio_size):
    _ = default_rng()
    
    # Générer une variable de base
    u0 = uniform.rvs(size=num_samples)
    
    # Générer les marges conditionnelles
    samples = []
    for _ in range(portfolio_size):
        u = uniform.rvs(size=num_samples)
        marginal = (u0 ** (-theta) - 1 + u ** (-theta)) ** (-1 / theta)
        samples.append(marginal)
    
    return np.array(samples).T

def gumbel_copula_multivariate(theta, num_samples, portfolio_size):
    """
    Génère des échantillons multivariés à partir d'une copule de Gumbel.

    Args:
        theta (float): Paramètre de dépendance de la copule de Gumbel (\( \theta \geq 1 \)).
        num_samples (int): Nombre d'échantillons à générer.
        portfolio_size (int): Dimensionnalité du portefeuille (nombre de prêts).

    Returns:
        ndarray: Échantillons multivariés de la copule de Gumbel.
    """
    if theta < 1:
        raise ValueError("Theta doit être supérieur ou égal à 1 pour la copule de Gumbel.")

    _ = default_rng()
    
    # Générer une variable de base
    u0 = uniform.rvs(size=num_samples)
    
    # Générer les marges conditionnelles
    samples = []
    for _ in range(portfolio_size):
        u = uniform.rvs(size=num_samples)
        marginal = np.exp(-((-np.log(u0)) ** theta + (-np.log(u)) ** theta) ** (1 / theta))
        samples.append(marginal)
    
    return np.array(samples).T


def plot_samples(samples, title):
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, edgecolor='k')
    plt.title(title)
    plt.xlabel('U1')
    plt.ylabel('U2')
    plt.grid()
    plt.show()
    

def transform_to_exponential(uniform_samples, lambda_rate=1):
    """Transform to real life distribution (exp for time to default)

    Args:
        uniform_samples (list): the sample to transform
        lambda_rate (int, optional): The parameter of the exp. Defaults to 1.

    Returns:
        _type_: the exp samples
    """
    return -np.log(1 - uniform_samples) / lambda_rate