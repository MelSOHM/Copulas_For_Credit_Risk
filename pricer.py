import utils_simulation as ut
import numpy as np

# Une seul période de temps 
# TODO : Pricer avec temps de défault

def pricing_CLO(defaults_matrix, portfolio, recovery_rate=0.4, credit_spreads=None, risk_free_rate=0.03):
    # Simulation des pertes agrégées (issues des défauts)
    losses = (1 - recovery_rate) * defaults_matrix @ portfolio["Loan_Amount"].values
    # Allocation des pertes aux tranches
    equity_losses, mezzanine_losses, senior_losses =  ut.allocate_losses_by_transches(defaults_matrix, portfolio)
    tranche_losses = {
        "Senior": senior_losses,
        "Mezzanine": mezzanine_losses,
        "Equity": equity_losses,
    } 
    # Pertes attendues pour chaque tranche
    expected_losses = {k: np.mean(v) for k, v in tranche_losses.items()}
    if credit_spreads == None:
        credit_spreads = { # en bp
            "Senior": 100,       
            "Mezzanine": 300,    
            "Équité": 1000,      
        }
    target_yields = {k: risk_free_rate + (v / 10000) for k, v in credit_spreads.items()}
    portfolio_total = portfolio["Loan_Amount"].sum()
    # Prix des tranches (basé sur les pertes actualisées)
    tranche_prices = {
        tranche: portfolio_total - calculate_present_value(losses, risk_free_rate) for tranche, losses in tranche_losses.items()
    } 
    # Calculer les rendements implicites pour chaque tranche
    implied_yields = {
        tranche: (1 - (loss / portfolio_total)) * (1 + target_yields[tranche]) - 1
        for tranche, loss in expected_losses.items()
    }
    
    
def calculate_present_value(losses, discount_rate, num_periods=1):
    """
    Calcule la valeur actuelle.
    """
    return np.sum(losses / ((1 + discount_rate) ** np.arange(1, num_periods + 1)))