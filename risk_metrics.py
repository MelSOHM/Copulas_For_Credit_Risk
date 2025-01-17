import numpy as np
import matplotlib.pyplot as plt

def calculate_var_and_es(losses, confidence_level=0.99):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) for a given loss distribution.
    
    Args:
        losses (numpy.ndarray): Array of simulated losses for a tranche.
        confidence_level (float): Confidence level for VaR and ES (default = 99%).

    Returns:
        tuple: (VaR, ES)
    """
    sorted_losses = np.sort(losses)
    var_index = int((1 - confidence_level) * len(sorted_losses))
    var = sorted_losses[var_index]
    es = sorted_losses[:var_index].mean()  # Average of losses beyond VaR
    return var, es

def tranche_risk_analysis(losses_per_period, tranche_limits, tranche_names, confidence_level=0.99):
    """
    Analyze the risk of each tranche by calculating VaR, ES, and plotting loss distributions.
    
    Args:
        losses_per_period (numpy.ndarray): Simulated losses for the portfolio (num_samples x num_periods).
        tranche_limits (list): List of tranche attachment points (e.g., [0.1, 0.3, 1.0]).
        tranche_names (list): Names of the tranches (e.g., ["Equity", "Mezzanine", "Senior"]).
        confidence_level (float): Confidence level for VaR and ES (default = 99%).

    Returns:
        dict: Risk metrics (VaR, ES) for each tranche.
    """
    total_losses = losses_per_period.sum(axis=2)  # Aggregate losses across periods

    tranche_metrics = {}
    cumulative_limits = np.cumsum(tranche_limits) * total_losses.max()  # Absolute limits

    for i, tranche_name in enumerate(tranche_names):
        # Define tranche loss exposure
        lower_limit = cumulative_limits[i - 1] if i > 0 else 0
        upper_limit = cumulative_limits[i]
        
        tranche_losses = np.clip(total_losses - lower_limit, 0, upper_limit - lower_limit)

        # Calculate VaR and ES
        var, es = calculate_var_and_es(tranche_losses, confidence_level)
        tranche_metrics[tranche_name] = {"VaR": var, "ES": es, "Losses": tranche_losses}

        # Plot loss distribution
        plt.hist(tranche_losses, bins=50, alpha=0.7, label=f"{tranche_name} Loss Distribution")
    
    plt.title("Loss Distributions by Tranche")
    plt.xlabel("Loss Amount")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    return tranche_metrics