import numpy as np
import pandas as pd
from sklearn.datasets import make_spd_matrix

# Parameters
num_loans = 50
seed = 42
np.random.seed(seed)  # For reproducibility

# Generate synthetic data
loan_ids = [f"Loan_{i+1}" for i in range(num_loans)]
loan_amounts = np.random.uniform(5e5, 2e6, num_loans)  # Loan amounts between $500k and $2M
maturities = np.random.randint(1, 10, num_loans)  # Maturities between 1 and 10 years
default_probabilities = np.random.uniform(0.01, 0.2, num_loans)  # Default probabilities between 1% and 20%
interest_rates = np.random.uniform(0.02, 0.1, num_loans)  # Interest rates between 2% and 10%

def generate_correlation_matrix(size):
    """
    Génère une matrice de corrélation symétrique et définie positive (PSD).
    
    Args:
        size (int): Taille de la matrice (nombre de variables).
    
    Returns:
        ndarray: Matrice de corrélation PSD.
    """
    # Étape 1 : Générer une matrice aléatoire
    random_matrix = np.random.uniform(-1, 1, (size, size))
    
    # Étape 2 : Rendre la matrice symétrique
    symmetric_matrix = (random_matrix + random_matrix.T) / 2
    
    # Étape 3 : Ajouter une diagonale dominante pour assurer la PSD
    # (optionnel si taille réduite ou pour renforcer la stabilité)
    for i in range(size):
        symmetric_matrix[i, i] = 1
    
    # Étape 4 : Utiliser la décomposition de Cholesky pour forcer la PSD
    eigvals, eigvecs = np.linalg.eigh(symmetric_matrix)
    eigvals[eigvals < 0] = 1e-1  # Remplacer les valeurs négatives par zéro
    psd_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Étape 5 : Normaliser en matrice de corrélation (valeurs dans [-1, 1])
    d = np.sqrt(np.diag(psd_matrix))
    corr_matrix = psd_matrix / np.outer(d, d)
    corr_matrix[np.diag_indices_from(corr_matrix)] = 1  # Diagonale à 1

    return corr_matrix

correlation_matrix = generate_correlation_matrix(num_loans)

# Create a DataFrame for the loan portfolio
loan_portfolio = pd.DataFrame({
    "Loan_ID": loan_ids,
    "Loan_Amount": loan_amounts,
    "Maturity_Years": maturities,
    "Default_Probability": default_probabilities,
    "Interest_Rate": interest_rates
})

# Save the data and correlation matrix to CSV for further use
loan_portfolio.to_csv("loan_portfolio.csv", index=False)
np.savetxt("correlation_matrix.csv", correlation_matrix, delimiter=",", fmt="%.3f")

print("Synthetic loan portfolio data and correlation matrix generated and saved.")