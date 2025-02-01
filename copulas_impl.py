import numpy as np
from scipy.stats import norm, multivariate_normal, gamma, uniform, levy_stable
from copulas.univariate import GaussianKDE
from copulas.bivariate import Clayton, Gumbel
import matplotlib.pyplot as plt
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

'''
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
'''

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



def clayton_copula_multivariate(theta, num_samples, portfolio_size):
    """Simule un échantillon de la copule de Clayton.
    
    Args:
        theta (float): Paramètre de dépendance de la copule de Gumbel (\( \theta \geq 0 \)).
        num_samples (int): Nombre d'échantillons à générer.
        portfolio_size (int): Dimensionnalité du portefeuille (nombre de prêts).

    Returns:
        ndarray: Échantillons multivariés de la copule de Gumbel. shape (num_samples, portfolio_size)
    """
    
    V = gamma.rvs(a=1/theta, scale=1, size=num_samples)  # Étape 1
    X = uniform.rvs(size=(num_samples, portfolio_size))  
    X = -np.log(X)  # Étapes 2/3 echantillon iid loi Exp(1) 
    U = (1 + (X / V[:, np.newaxis])) ** (-1/theta)  # Étape 4 num_samples echantillons independants d'une copule de clayton de 
    # dimension portfolio_size
    return U

def gumbel_copula_multivariate(theta, num_samples, portfolio_size):
    """Simule un échantillon de la copule de Gumbel.
    
    Args:
        theta (float): Paramètre de dépendance de la copule de Gumbel (\( \theta \geq 1 \)).
        num_samples (int): Nombre d'échantillons à générer.
        portfolio_size (int): Dimensionnalité du portefeuille (nombre de prêts).

    Returns:
        ndarray: Échantillons multivariés de la copule de Gumbel. shape (num_samples, portfolio_size)
    """
    if theta < 1:
        raise ValueError("Theta doit être supérieur ou égal à 1 pour la copule de Gumbel.")    
    alpha = 1 / theta
    # Étape 1: Générer U1, U2 ~ U(0,1)
    U = np.random.uniform(0, 1, (num_samples,2))
    E2 = -np.log(U[:,0])
    U1 = np.pi * (U[:,1] - 0.5)
    # Étape 4: Calculer a = 0.5π + U1
    a = (0.5 * np.pi + U1) * alpha
    # Étape 5: Calculer s = sin(a)
    s = np.sin(a)
    # Étape 6: Calculer c = cos(U1 - a)
    c = np.cos(U1) ** (1/alpha)
    V = (np.cos(U1 - a) / E2) ** ((1-alpha)/alpha) * s / c
    # Générer X1, ..., Xd ~ U(0,1)
    X = np.random.uniform(0, 1, (num_samples, portfolio_size))
    # Transformer X en X ~ Exp(1)
    X = -np.log(X)
    U = np.exp(-(X/V[:,None])**(1/theta))
    return U

