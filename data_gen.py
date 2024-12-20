import numpy as np
import pandas as pd

# Parameters
num_loans = 50
np.random.seed(42)  # For reproducibility

# Generate synthetic data
loan_ids = [f"Loan_{i+1}" for i in range(num_loans)]
loan_amounts = np.random.uniform(5e5, 2e6, num_loans)  # Loan amounts between $500k and $2M
maturities = np.random.randint(1, 10, num_loans)  # Maturities between 1 and 10 years
default_probabilities = np.random.uniform(0.01, 0.2, num_loans)  # Default probabilities between 1% and 20%

# Generate correlation matrix
correlation_matrix = np.random.uniform(0.1, 0.5, (num_loans, num_loans))
np.fill_diagonal(correlation_matrix, 1)  # Diagonal elements are 1 (perfect correlation with itself)
correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Symmetrize

# Create a DataFrame for the loan portfolio
loan_portfolio = pd.DataFrame({
    "Loan_ID": loan_ids,
    "Loan_Amount": loan_amounts,
    "Maturity_Years": maturities,
    "Default_Probability": default_probabilities
})

# Save the data and correlation matrix to CSV for further use
loan_portfolio.to_csv("loan_portfolio.csv", index=False)
np.savetxt("correlation_matrix.csv", correlation_matrix, delimiter=",", fmt="%.3f")

print("Synthetic loan portfolio data and correlation matrix generated and saved.")
