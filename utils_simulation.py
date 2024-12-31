import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_portfolio() -> pd.DataFrame:
    """Read the generated porfolio (using data gen)

    Returns:
        _type_: dataframe
    """
    return pd.read_csv("loan_portfolio.csv")


def simulate_defaults(samples_uniform, default_probabilities):
    """
    Simule les défauts dans le portefeuille basé sur des échantillons uniformes.

    Args:
        samples_uniform (ndarray): Échantillons uniformes corrélés générés par la copule.
        default_probabilities (ndarray): Probabilités de défaut individuelles pour chaque prêt.

    Returns:
        ndarray: Matrice binaire des défauts (1 = défaut, 0 = pas de défaut).
    """
    return (samples_uniform < default_probabilities).astype(int)

def print_summary(defaults_matrix, portfolio, recovery_rate = 0.4):
    """Print some analytics about the simulation

    Args:
        defaults_matrix (_pd.DataFrame_): the default matrix from the simmulation
        portfolio (_pd.DataFrame_): The Portfolio for the simmulation
        recovery_rate (float): recouvrement
    """
    default_rates = defaults_matrix.mean(axis=1)
    print("Résumé des scénarios de défauts :")
    print(f"Taux moyen de défaut : {default_rates.mean():.2%}")
    print(f"Taux minimum de défaut : {default_rates.min():.2%}")
    print(f"Taux maximum de défaut : {default_rates.max():.2%}")
    
    plt.hist(default_rates, bins=20, edgecolor="k", alpha=0.7)
    plt.title("Distribution des Taux de Défaut par Scénario")
    plt.xlabel("Taux de Défaut")
    plt.ylabel("Fréquence")
    plt.show()

    threshold = 0.3
    extreme_scenarios = (default_rates > threshold).mean()
    print(f"Probabilité de scénarios avec plus de {threshold*100:.0f}% de défauts : {extreme_scenarios:.2%}")
    
    
    default_correlations = np.corrcoef(defaults_matrix.T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(default_correlations, cmap="coolwarm", center=0, annot=False)
    plt.title("Corrélations entre les Défauts des Prêts")
    plt.show()
    
    losses = (1 - recovery_rate) * defaults_matrix @ portfolio["Loan_Amount"].values

    # Résumé des pertes
    print(f"Pertes moyennes : ${losses.mean():,.0f}")
    print(f"Pertes maximales : ${losses.max():,.0f}")
    print(f"Pertes minimales : ${losses.min():,.0f}")
    
    portfolio["Tranche"] = pd.cut(portfolio["Loan_Amount"], bins=[0, 1000000, 1500000, 2000000], labels=["0-1M", "1-1.5M", "1.5-2M"])
    portfolio["Average_Default_Contribution"] = defaults_matrix.mean(axis=0)
    # Calcul des taux de défaut moyens par tranche
    default_rates_by_tranche = portfolio.groupby("Tranche")["Average_Default_Contribution"].mean()

    print("Taux de défaut moyen par tranche :\n", default_rates_by_tranche)
    

def allocate_losses_by_transches(defaults_matrix, portfolio, recovery_rate = 0.4,
                                 senior_attachment= 0.3, mezz_attachment= 0.1, percentage=True):
    losses = (1 - recovery_rate) * defaults_matrix @ portfolio["Loan_Amount"].values
    portfolio_total = portfolio["Loan_Amount"].sum()

    return apply_waterwall(losses, portfolio_total, senior_attachment, mezz_attachment, percentage)


def apply_waterwall(losses, portfolio_total, 
                    senior_attachment= 0.3, mezz_attachment= 0.1, percentage=True):
    senior_limit = senior_attachment * portfolio_total
    mezzanine_limit = mezz_attachment * portfolio_total
    
    equity_losses = np.where(losses > mezzanine_limit, mezzanine_limit, losses)
    remaining_losses = losses - equity_losses
    
    mezzanine_losses = np.where(remaining_losses > senior_limit - mezzanine_limit, senior_limit - mezzanine_limit, remaining_losses)
    remaining_losses = remaining_losses - mezzanine_losses

    senior_losses = remaining_losses
    
    if percentage:
        equity_losses = np.round(equity_losses/mezzanine_limit*100,4)
        mezzanine_losses = np.round(mezzanine_losses/(senior_limit - mezzanine_limit)*100,4)
        senior_losses = np.round(equity_losses/(portfolio_total-senior_limit)*100,4)
        
    return equity_losses, mezzanine_losses, senior_losses


def simmulate_losses_tranche(equity_losses, mezzanine_losses, senior_losses):
    summary = {
        "Tranche": ["Senior", "Mezzanine", "Equity"],
        "Pertes Moyennes": [np.mean(senior_losses), np.mean(mezzanine_losses), np.mean(equity_losses)],
        "Pertes Maximales": [np.max(senior_losses), np.max(mezzanine_losses), np.max(equity_losses)],
        "Pertes Minimales": [np.min(senior_losses), np.min(mezzanine_losses), np.min(equity_losses)],
    }
    summary_df = pd.DataFrame(summary)
    print(summary_df)
    
    plt.boxplot([senior_losses, mezzanine_losses, equity_losses], labels=["Senior", "Mezzanine", "Equity"])
    plt.title("Distribution des Pertes par Tranche")
    plt.ylabel("Pertes (%)")
    plt.grid(True)
    plt.show()