import numpy as np
import matplotlib.pyplot as plt
from pricer import pricing_CLO_multi_periode_gaussian
import seaborn as sns
import pandas as pd

# Scénarios de stress
scenarios = [
    {"default_prob_multiplier": 1.2, "correlation_boost": 0.1, "recovery_rate": 0.3},
    {"default_prob_multiplier": 1.5, "correlation_boost": 0.2, "recovery_rate": 0.2},
]

def calculate_var_and_es(losses, confidence_level=0.99):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) for a given loss distribution.

    Args:
        losses (numpy.ndarray): Array of aggregated losses (num_samples,).
        confidence_level (float): Confidence level for VaR and ES (default = 99%).

    Returns:
        tuple: (VaR, ES) aggregated across all periods.
    """
    sorted_losses = np.sort(losses)
    var_index = int((1 - confidence_level) * len(sorted_losses))
    var = sorted_losses[var_index]
    es = sorted_losses[:var_index].mean()  # Expected Shortfall (average of worst cases)
    return var, es

def tranche_risk_analysis(loss_dict, tranche_limits, tranche_names, portfolio, initial_investments, confidence_level=0.99):
    """
    Compute risk metrics (VaR, Expected Shortfall) for each tranche based on simulated losses.
    Generates clear visualizations for better understanding.

    Args:
        loss_dict (numpy.ndarray): Simulated losses per loan over time (samples, loans, time).
        tranche_limits (list): List of tranche attachment points (e.g., [0.1, 0.3, 1.0]).
        tranche_names (list): Names of the tranches (e.g., ["Equity", "Mezzanine", "Senior"]).
        portfolio (pd.DataFrame): Portfolio containing Loan_Amount and other loan characteristics.
        initial_investments (dict): Initial investment per tranche.
        confidence_level (float): Confidence level for VaR and ES (default = 99%).

    Returns:
        dict: Risk metrics (VaR, ES, Loss Ratios) for each tranche.
    """
    num_samples = loss_dict.shape[0]

    # Step 1: Aggregate total losses per sample across all loans and time periods
    total_losses = loss_dict.sum(axis=(1, 2))  # Shape: (num_samples,)

    # Step 2: Compute absolute tranche limits based on portfolio total size
    portfolio_total = portfolio["Loan_Amount"].sum()
    cumulative_limits = np.cumsum(tranche_limits) * portfolio_total

    tranche_metrics = {}
    all_losses_df = pd.DataFrame()  # DataFrame pour stocker les pertes par tranche

    for i, tranche_name in enumerate(tranche_names):
        lower_limit = cumulative_limits[i - 1] if i > 0 else 0
        upper_limit = cumulative_limits[i]

        # Tranche-specific losses: only take losses above lower limit but capped at upper limit
        tranche_losses = np.clip(total_losses - lower_limit, 0, upper_limit - lower_limit)

        # Compute Value at Risk (VaR) and Expected Shortfall (ES)
        var, es = calculate_var_and_es(tranche_losses, confidence_level)

        # Récupérer l'investissement initial pour la tranche
        initial_investment = initial_investments.get(tranche_name, 1)  # Éviter division par 0

        # Calculer le ratio de pertes par rapport à l’investissement initial
        loss_ratio = (tranche_losses / initial_investment) * 100  # En pourcentage

        # Stocker les résultats
        tranche_metrics[tranche_name] = {
            "VaR": var,
            "ES": es,
            "Loss Ratio (%)": np.mean(loss_ratio),
            "Initial Investment": initial_investment,
            "Losses": tranche_losses
        }

        # Stocker les pertes dans un DataFrame pour les comparaisons
        tranche_data = pd.DataFrame({
            "Losses": tranche_losses,
            "Tranche": tranche_name,
            "Loss Ratio (%)": loss_ratio
        })
        all_losses_df = pd.concat([all_losses_df, tranche_data], ignore_index=True)

        ### PLOT 1: Histogramme de la distribution des pertes pour cette tranche
        plt.figure(figsize=(8, 5))
        sns.histplot(tranche_losses, bins=50, kde=True, color="blue", alpha=0.6)
        plt.axvline(var, color="red", linestyle="--", label=f"VaR (99%): {var:,.0f} €")
        plt.axvline(es, color="purple", linestyle="--", label=f"Expected Shortfall: {es:,.0f} €")
        plt.title(f"Loss Distribution - {tranche_name}")
        plt.xlabel("Loss Amount (€)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    ### PLOT 2: Comparaison des distributions de pertes entre tranches
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data=all_losses_df, x="Losses", hue="Tranche", bins=50, kde=True, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, title="Tranche", loc="upper right")
    plt.title("Comparison of Loss Distributions Across Tranches")
    plt.xlabel("Loss Amount (€)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    ### PLOT 3: Boxplot des pertes par tranche
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Tranche", y="Losses", data=all_losses_df, palette="Set2")
    plt.title("Loss Distribution by Tranche")
    plt.xlabel("Tranche")
    plt.ylabel("Loss Amount (€)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    ### PLOT 4: Comparaison "Investissement vs Pertes"
    plt.figure(figsize=(8, 5))
    investment_vs_losses = pd.DataFrame(tranche_metrics).T[["Initial Investment", "VaR", "ES"]]
    investment_vs_losses.plot(kind="bar", figsize=(10, 6), colormap="coolwarm")
    plt.title("Investment vs Risk Metrics (VaR & ES)")
    plt.ylabel("Amount (€)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)
    plt.legend(["Initial Investment", "VaR (99%)", "Expected Shortfall"])
    plt.show()

    ### Affichage des résultats sous forme de tableau
    tranche_summary = pd.DataFrame.from_dict(tranche_metrics, orient='index', columns=["VaR", "ES", "Loss Ratio (%)", "Initial Investment"])
    tranche_summary["VaR"] = tranche_summary["VaR"].apply(lambda x: f"{x:,.0f} €")
    tranche_summary["ES"] = tranche_summary["ES"].apply(lambda x: f"{x:,.0f} €")
    tranche_summary["Loss Ratio (%)"] = tranche_summary["Loss Ratio (%)"].apply(lambda x: f"{x:.2f} %")
    tranche_summary["Initial Investment"] = tranche_summary["Initial Investment"].apply(lambda x: f"{x:,.0f} €")

    print("\n🔹 **Tranche Risk Summary (VaR, Expected Shortfall & Loss Ratios)** 🔹")
    print(tranche_summary.to_markdown())

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