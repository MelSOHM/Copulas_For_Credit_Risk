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

def apply_waterwall_fluxes(losses, principal_payments, interest_payments, spread, portfolio_total, 
                           senior_attachment= 0.3, mezz_attachment= 0.1):
    
    senior_limit = senior_attachment * portfolio_total
    mezzanine_limit = mezz_attachment * portfolio_total
    
    # Losses 
    equity_losses = np.where(losses > mezzanine_limit, mezzanine_limit, losses)
    remaining_losses = losses - equity_losses
    mezzanine_losses = np.where(remaining_losses > senior_limit - mezzanine_limit, senior_limit - mezzanine_limit, remaining_losses)
    remaining_losses = remaining_losses - mezzanine_losses
    senior_losses = remaining_losses
    
    #Cash
    senior_cash = min(1-senior_attachment)
            
    return equity_losses, mezzanine_losses, senior_losses

def calculate_cdo_cashflows_with_limits(principal_payments, interest_payments, losses, 
                                        tranche_limits, tranche_rates, risk_free_rate):
    """
    Calcule les flux de trésorerie pour les tranches d'un CDO en tenant compte des pertes, paiements,
    et des soldes cumulés des tranches.
    
    Parameters:
        principal_payments (numpy.ndarray): Tableau 3D (samples, loans, times) des paiements de principal.
        interest_payments (numpy.ndarray): Tableau 3D (samples, loans, times) des paiements d'intérêts.
        losses (numpy.ndarray): Tableau 3D (samples, loans, times) des pertes.
        tranche_limits (list): Limites cumulées des tranches [Equity %, Mezzanine %, Senior %].
        tranche_rates (list): Rémunérations des tranches (Mezzanine, Senior).
        risk_free_rate (float): Taux sans risque pour actualiser les flux.

    Returns:
        dict: Un dictionnaire avec les flux nets pour chaque tranche sous forme de tableau (samples, times).
    """
    # Combiner les flux nets
    net_flows = principal_payments + interest_payments - losses  # (samples, loans, times)
    
    # Agréger les flux par temps pour chaque sample
    total_flows = np.sum(net_flows, axis=1)  # (samples, times)
    total_losses = np.sum(losses, axis=1)  # (samples, times)
    
    # Initialiser les résultats
    tranche_results = {
        "Equity": np.zeros_like(total_flows),
        "Mezzanine": np.zeros_like(total_flows),
        "Senior": np.zeros_like(total_flows)
    }
    
    mezzanine_rate, senior_rate = tranche_rates + risk_free_rate

    # Soldes cumulés pour chaque tranche
    tranche_balances = {
        "Equity": np.full(total_flows.shape[0], tranche_limits[0]),  # 10% initial pour Equity
        "Mezzanine": np.full(total_flows.shape[0], tranche_limits[1]),  # 20% initial pour Mezzanine
        "Senior": np.full(total_flows.shape[0], tranche_limits[2])   # 70% initial pour Senior
    }

        # Soldes cumulés pour chaque tranche
    tranche_expected_perf = {
        "Equity": np.full(total_flows.shape[0], tranche_limits[0]),  # 10% initial pour Equity
        "Mezzanine": np.full(total_flows.shape[0], tranche_limits[1]*(1+mezzanine_rate)),  # 20% initial pour Mezzanine
        "Senior": np.full(total_flows.shape[0], tranche_limits[2]*(1+senior_rate))   # 70% initial pour Senior
    }

    # Répartir les pertes et les paiements par tranche
    for t in range(total_flows.shape[1]):  # Pour chaque période
        for sample in range(total_flows.shape[0]):  # Pour chaque échantillon
            remaining_loss = total_losses[sample, t]
            remaining_payment = total_flows[sample, t]

            # Allouer les pertes
            # Pertes sur Equity
            if tranche_balances["Equity"][sample] > 0:
                equity_loss = min(remaining_loss, tranche_balances["Equity"][sample])
                tranche_results["Equity"][sample, t] -= equity_loss
                tranche_balances["Equity"][sample] -= equity_loss
                remaining_loss -= equity_loss

            # Pertes sur Mezzanine
            if tranche_balances["Mezzanine"][sample] > 0:
                mezzanine_loss = min(remaining_loss, tranche_balances["Mezzanine"][sample])
                tranche_results["Mezzanine"][sample, t] -= mezzanine_loss
                tranche_balances["Mezzanine"][sample] -= mezzanine_loss
                remaining_loss -= mezzanine_loss
                
            # Pertes sur Senior
            senior_loss = min(remaining_loss, tranche_balances["Senior"][sample])
            tranche_results["Senior"][sample, t] -= senior_loss
            tranche_balances["Senior"][sample] -= senior_loss
            remaining_loss -= senior_loss
            
            assert remaining_loss == 0 # Il n'est plus censé y avoir de perte a épongé
            # Allouer les paiements
            # Paiements sur Senior
            if tranche_expected_perf["Senior"][sample] > 0:
                senior_payment = min(remaining_payment, tranche_expected_perf["Senior"][sample])
                tranche_results["Senior"][sample, t] += senior_payment
                tranche_expected_perf["Senior"][sample] -= senior_payment
                remaining_payment -= senior_payment

            # Paiements sur Mezzanine
            if tranche_expected_perf["Mezzanine"][sample] > 0:
                mezzanine_payment = min(remaining_payment, tranche_expected_perf["Mezzanine"][sample])
                tranche_results["Mezzanine"][sample, t] += mezzanine_payment
                tranche_expected_perf["Mezzanine"][sample] -= mezzanine_payment
                remaining_payment -= mezzanine_payment

            # Paiements sur Equity
            equity_payment = remaining_payment  # Ce qui reste va à Equity
            tranche_results["Equity"][sample, t] += equity_payment
            remaining_payment -= equity_payment
                
            assert remaining_payment == 0

    return tranche_results, tranche_limits, [tranche_limits[0],tranche_limits[1]*(1+mezzanine_rate),tranche_limits[2]*(1+senior_rate)]

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