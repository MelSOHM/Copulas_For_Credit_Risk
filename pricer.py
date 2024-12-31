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
                                       risk_free_rate=0.03, num_samples=100):
    
    num_periods = portfolio["Maturity_Years"].max()
    loan_states = np.zeros((num_samples, len(portfolio), num_periods))  # 0 = non défaut, 1 = défaut
    annual_default_probabilities = portfolio["Default_Probability"].values  # Probabilités de défaut initiales (à un an)

    correlation_matrix = conversion_to_array(correlation_matrix)
    active_correlation_matrix = correlation_matrix.copy()  # Matrice de corrélation initiale
    
    # Simuler les dépendances corrélées pour chaque période
    for t in range(num_periods):
        if t == 0:
            # Première période : directement utiliser les probabilités initiales
            correlated_samples = cop.gaussian_copula_sample(active_correlation_matrix, num_samples)
            loan_states[:, :, t] = (correlated_samples < annual_default_probabilities).astype(int)
        else:
            # Périodes suivantes : conditionner sur l'état précédent
            previous_defaults = loan_states[:, :, t - 1]
            active_loans = previous_defaults == 0  # Prêts non en défaut

            # Mettre à jour la matrice de corrélation pour les prêts actifs
            active_indices = np.where(np.any(active_loans == 0, axis=0))[0]  # Prêts actifs
            if len(active_indices) > 1:  # Recalculer si des prêts sont encore actifs
                active_correlation_matrix = correlation_matrix[np.ix_(active_indices, active_indices)]
                
                # Générer les échantillons corrélés pour les prêts actifs
                correlated_samples = cop.gaussian_copula_sample(active_correlation_matrix, num_samples)
                
                # Appliquer les probabilités de défaut conditionnelles
                conditional_defaults = (correlated_samples < annual_default_probabilities[active_indices]).astype(int)
                loan_states[:, active_indices, t] = conditional_defaults
            else:
                break  # Arrêter si moins de 2 prêts actifs

    # Calculer les pertes
    loan_amounts = portfolio["Loan_Amount"].values
    portfolio_total = portfolio["Loan_Amount"].sum()
    losses_per_period = (1 - recovery_rate) * (loan_states * loan_amounts[None, :, None])
    # Cascade des pertes pour chaque période
    equity_losses_list, mezzanine_losses_list, senior_losses_list = [], [], []
    for t in range(num_periods):
        period_loss = np.nansum(losses_per_period[:, :, t], axis=1)  # Pertes totales pour la période
        equity_losses, mezzanine_losses, senior_losses = ut.apply_waterwall(period_loss, 
                                                                            portfolio_total, 
                                                                            percentage=False)
        equity_losses_list.append(equity_losses)
        mezzanine_losses_list.append(mezzanine_losses)
        senior_losses_list.append(senior_losses)
        
    # Calculer les prix actualisés des tranches
    tranche_prices = {
        "Senior": calculate_present_value(senior_losses_list, risk_free_rate).mean(),
        "Mezzanine": calculate_present_value(mezzanine_losses_list, risk_free_rate).mean(),
        "Equity": calculate_present_value(equity_losses_list, risk_free_rate).mean(),
    }

    return losses_per_period, tranche_prices

    
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