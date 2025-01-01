import utils_simulation as ut
import numpy as np
import copulas_impl as cop

# def pricing_CLO(defaults_matrix, portfolio, recovery_rate=0.4, credit_spreads=None, risk_free_rate=0.03):
    
#     # Allocation des pertes aux tranches
#     equity_losses, mezzanine_losses, senior_losses =  ut.allocate_losses_by_transches(defaults_matrix, 
#                                                                                       portfolio,
#                                                                                       recovery_rate=recovery_rate)
#     tranche_losses = {
#         "Senior": senior_losses,
#         "Mezzanine": mezzanine_losses,
#         "Equity": equity_losses,
#     } 
#     # Pertes attendues pour chaque tranche
#     expected_losses = {k: np.mean(v) for k, v in tranche_losses.items()}
#     if credit_spreads == None:
#         credit_spreads = { # en bp
#             "Senior": 100,       
#             "Mezzanine": 300,    
#             "Équité": 1000,      
#         }
#     target_yields = {k: risk_free_rate + (v / 10000) for k, v in credit_spreads.items()}
#     portfolio_total = portfolio["Loan_Amount"].sum()
#     # Prix des tranches (basé sur les pertes actualisées)
#     tranche_prices = {
#         tranche: portfolio_total - calculate_present_value(losses, risk_free_rate) for tranche, losses in tranche_losses.items()
#     } 
#     # Calculer les rendements implicites pour chaque tranche
#     implied_yields = {
#         tranche: (1 - (loss / portfolio_total)) * (1 + target_yields[tranche]) - 1
#         for tranche, loss in expected_losses.items()
#     }
    
def pricing_CLO_multi_periode_gaussian(correlation_matrix, portfolio, recovery_rate=0.4, 
                                       tranche_spreads=None, risk_free_rate=0.03, num_samples=100,
                                       senior_attachment= 0.3, mezz_attachment= 0.1):
    
    num_periods = portfolio["Maturity_Years"].max()
    loan_states = np.zeros((num_samples, len(portfolio), num_periods))  # 0 = non défaut, 1 = défaut 
    annual_default_probabilities = portfolio["Default_Probability"].values  # Probabilités de défaut initiales (à un an)

    tranche_spreads = tranche_spreads or {"Equity": 0.1, "Mezzanine": 0.03, "Senior": 0.01}  # Par défaut en %
    correlation_matrix = conversion_to_array(correlation_matrix)
    active_correlation_matrix = correlation_matrix.copy()  # Matrice de corrélation initiale
    
    for t in range(num_periods):
        if t == 0:
            # Première période : directement utiliser les probabilités initiales
            correlated_samples = cop.gaussian_copula_sample(active_correlation_matrix, num_samples)
            loan_states[:, :, t] = (correlated_samples < annual_default_probabilities).astype(int)
        else:
            # Identifier les prêts actifs (non en défaut et dans leur période de maturité)
            no_default = np.array(loan_states[:, :, t - 1]==0, dtype=bool)
            not_matured = np.array(portfolio["Maturity_Years"].values[np.newaxis, :] > t, dtype=bool)
            active_loans = (no_default & not_matured).astype(int) # Non en défaut et t < maturité 
            # Simuler uniquement pour les prêts actifs
            for sample_idx in range(active_loans.shape[0]):
                active_indices = np.where(active_loans[sample_idx] == 1)[0]
                if len(active_indices) > 0:
                    active_correlation_matrix = correlation_matrix[np.ix_(active_indices, active_indices)]
                    correlated_samples = cop.gaussian_copula_sample(active_correlation_matrix, 1)
                    # Simuler les défauts conditionnels
                    conditional_defaults = (correlated_samples < annual_default_probabilities[active_indices]).astype(int)
                    loan_states[sample_idx, active_indices, t] = conditional_defaults
                
    # Calculer les pertes
    loan_amounts = portfolio["Loan_Amount"].values
    portfolio_total = portfolio["Loan_Amount"].sum()
    losses_per_period = (1 - recovery_rate) * (loan_states * loan_amounts[None, :, None])
    
    # Calcul des cash flows positifs
    principal_payments = np.zeros_like(loan_states)  # Initialiser à 0 pour toutes les périodes
    interest_payments = np.zeros_like(loan_states, dtype=float)

    for i, loan in enumerate(portfolio.itertuples()):
        maturity_period = int(loan.Maturity_Years) - 1  # Dernière période de maturité
        
        for t in range(num_periods):
            if t == maturity_period:
                # Paiement du principal uniquement à maturité si pas en défaut
                for sample in range(num_samples):
                    if np.all(loan_states[sample, i, :maturity_period + 1] == 0):  # Pas de défaut jusqu'à maturité
                        principal_payments[sample, i, t] = loan.Loan_Amount
                        interest_payments[sample, i, t] = loan.Loan_Amount * loan.Interest_Rate
            elif t < maturity_period:
                # Paiement des intérêts pour les périodes actives
                for sample in range(num_samples):
                    if np.all(loan_states[sample, i, :t + 1] == 0):  # Actif dans cette période
                        interest_payments[sample, i, t] = loan.Loan_Amount * loan.Interest_Rate
    
    cash_flows = principal_payments + interest_payments # Flux totaux (principal + intérêts)
    net_cash_flows = cash_flows - losses_per_period # Flux nets par période
    
    tranche_limits = portfolio_total*np.array([mezz_attachment, senior_attachment-mezz_attachment, 1-senior_attachment])
    tranche_prices = ut.calculate_cdo_cashflows_with_limits(principal_payments, 
                                                            interest_payments, 
                                                            losses_per_period, 
                                                            tranche_limits, 
                                                            tranche_rates=np.array(list(tranche_spreads.values())), 
                                                            risk_free_rate=risk_free_rate)
    tranche_prices = {
        "Senior": tranche_prices["Senior"].sum(axis=1).mean(),
        "Mezzanine": tranche_prices["Mezzanine"].sum(axis=1).mean(),
        "Equity": tranche_prices["Equity"].sum(axis=1).mean(),
    }

    return net_cash_flows, tranche_prices, interest_payments, principal_payments, loan_states, losses_per_period

    
def calculate_present_value(losses, discount_rate, num_periods=1):
    """
    Calcule la valeur actuelle.
    """
    return np.sum(losses / ((1 + discount_rate) ** np.arange(1, num_periods + 1)), axis=0) # sums sur les périodes

def conversion_to_array(df):
    try:
        return df.to_numpy()
    except:
        return df