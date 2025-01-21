import numpy as np
import matplotlib.pyplot as plt
from pricer import pricing_CLO_multi_periode_gaussian

# Scénarios de stress
scenarios = [
    {"default_prob_multiplier": 1.2, "correlation_boost": 0.1, "recovery_rate": 0.3},
    {"default_prob_multiplier": 1.5, "correlation_boost": 0.2, "recovery_rate": 0.2},
]

def calculate_var_and_es(losses, confidence_level=0.99):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) for aggregated losses.
    
    Args:
        losses (numpy.ndarray): Array of losses with shape (samples, loans, time).
        confidence_level (float): Confidence level for VaR and ES (default = 99%).

    Returns:
        tuple: (VaR, ES) aggregated across all loans and time.
    """
    # Aggregate losses across loans and time for each sample
    aggregated_losses = losses.sum(axis=1).sum(axis=1)  # Shape: (samples,)

    # Sort and calculate VaR/ES
    sorted_losses = np.sort(aggregated_losses)
    var_index = int((1 - confidence_level) * len(sorted_losses))
    var = sorted_losses[var_index]
    es = sorted_losses[:var_index].mean()  # Average of losses beyond VaR
    return var, es


def tranche_risk_analysis(losses_dict, tranche_limits, tranche_names, confidence_level=0.99):
    """
    Analyze the risk of each tranche by calculating VaR, ES, and plotting loss distributions.

    Args:
        losses_dict (numpy.ndarray): Array of losses with shape (samples, loans, time).
        tranche_limits (list): List of tranche attachment points (e.g., [0.1, 0.3, 1.0]).
        tranche_names (list): Names of the tranches (e.g., ["Equity", "Mezzanine", "Senior"]).
        confidence_level (float): Confidence level for VaR and ES (default = 99%).

    Returns:
        dict: Risk metrics (VaR, ES) for each tranche.
    """
    # Aggregate losses across loans and time for each sample
    total_losses = losses_dict.sum(axis=(1, 2))  # Shape: (samples,)

    # Calculate absolute tranche limits
    portfolio_total = total_losses.max()  # Assume the portfolio total is the max possible loss
    cumulative_limits = np.cumsum(tranche_limits) * portfolio_total

    tranche_metrics = {}

    for i, tranche_name in enumerate(tranche_names):
        # Define tranche loss exposure
        lower_limit = cumulative_limits[i - 1] if i > 0 else 0
        upper_limit = cumulative_limits[i]

        # Tranche-specific losses
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

def stress_testing_gaussian(scenario: dict, portfolio, correlation_matrix):
    stressed_portfolio = portfolio.copy()
    stressed_portfolio["Default_Probability"] *= scenario["default_prob_multiplier"]
    stressed_correlation_matrix = correlation_matrix + scenario["correlation_boost"]
    stressed_correlation_matrix = np.clip(stressed_correlation_matrix, 0, 1)  # Éviter des corrélations > 1
    
    result = pricing_CLO_multi_periode_gaussian(
        stressed_correlation_matrix,
        stressed_portfolio,
        recovery_rate=scenario["recovery_rate"],
        num_samples=1000
    )
    
    # Calculer les métriques de risque
    net_cash_flows, tranche_prices, _, _, _, losses_per_period, _, _ = result
    tranche_metrics = tranche_risk_analysis(
        losses_per_period=losses_per_period,
        tranche_limits=[0.1, 0.2, 0.7],
        tranche_names=["Equity", "Mezzanine", "Senior"]
    )