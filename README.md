# Copulas For Credit Risk

![Static Badge](https://img.shields.io/badge/Mission-Credit_derivatives_pricing-purple)
![Static Badge](https://img.shields.io/badge/Language-Python-green)
![Static Badge](https://img.shields.io/badge/Notice-attention_les_yeux-red)
<br />

Description of the project: You are tasked with structuring a collateralized loan obligation (CLO) comprising approximately 50 loans. To cater to varying investor risk preferences, the CLO is divided into three tranches: 0-10%, 10-30%, and 30-100%. To evaluate the risk and determine the pricing of each tranche, you decide to compare two models: one based on a Gaussian copula and the other on an Archimedean copula.

> [!NOTE]
>
> - Date de rendu: Dimanche 2 Février
> - Soutenances: entre le 3 et le 14 Février

## TO DO

1. Perform a numerical comparison of Gaussian and Archimedean copulas.
2. Develop a pricing tool for CLO tranches using both copula approaches.
3. Apply the tool to the given infrastructure loans portfolio.
4. Analyze the strengths and weaknesses of each model, particularly in the context of infrastructure loans.

## Ideas:

- **Data Preparation**: Gather and preprocess loan data, including loan amounts, maturities, default probabilities, and correlations. Ensure the dataset is clean and representative of infrastructure loans. ✅

- **Model Implementation**: Implement both Gaussian and Archimedean copulas in Python, using libraries like `scipy`, `copulas`, or `statsmodels` for efficient computation. ✅

- **Portfolio Simulation**: Simulate default scenarios for the portfolio of 50 loans, using the chosen copula models to generate correlated defaults. ✅

- **Tranche Structuring**: Define the cash flow waterfall for the CLO tranches (0-10%, 10-30%, 30-100%) and determine how losses are allocated across the tranches. ✅

- **Pricing Framework**: Develop a pricing mechanism for the CLO tranches based on expected losses, investor risk preferences, and market conditions (e.g., interest rates, recovery rates). Monte-Carlo pricing using the copulas ? *(still need to do for clayton and gumbell, documentation to be written)*

- **Risk Analysis**: Evaluate the tail risk and sensitivity of each tranche under different economic scenarios using stress testing and scenario analysis.

- **Comparison Metrics**: Define clear metrics (e.g., VaR, expected shortfall, tranche loss distributions) to compare the performance of Gaussian and Archimedean copulas.

- **Visualization**: Create visualizations to represent default correlations, tranche loss distributions, and risk profiles for different copula models. 2D plot of samples drawn from the 2 different copulas.

- **Documentation and Automation**: Develop a detailed user guide for the pricing tool, including examples of how to use it. Automate repetitive tasks for efficiency.

- **Insights and Recommendations**: Summarize findings on the suitability of each copula model for pricing CLOs and provide recommendations for practical use cases in infrastructure loans.

- **Analyse the results and test on insightful data**

- **Write the report**

## Documentation:

# Documentation : Fonction de Pricing CLO avec Copules Gaussiennes

Cette fonction implémente un modèle de pricer multi-périodes pour les Collateralized Loan Obligations (CLOs) en utilisant des copules gaussiennes pour simuler les corrélations de défaut entre les prêts d'un portefeuille. Elle permet de calculer les pertes, les flux de trésorerie (cash flows) positifs et négatifs, et d'estimer les prix des tranches (Senior, Mezzanine et Equity) sur plusieurs périodes.

---

## **Description de la Méthodologie**

### **1. Simulation des défauts avec une copule gaussienne**
La fonction commence par simuler les défauts des prêts dans le portefeuille en utilisant une copule gaussienne pour capturer les dépendances entre les défauts.

- **Étape initiale** : Les probabilités de défaut annuelles des prêts sont utilisées pour la première période.
- **Étapes suivantes** : Les prêts actifs (non en défaut et encore dans leur période de maturité) sont identifiés. Des échantillons corrélés sont générés pour ces prêts à chaque période suivante.
- **Conditionnalité** : Les défauts pour une période donnée sont simulés uniquement pour les prêts actifs, en tenant compte des corrélations conditionnelles définies par la copule.

### **2. Identification des prêts actifs**
Un prêt est considéré comme actif si :
- Il n'est pas en défaut pour toutes les périodes précédentes.
- Sa période de maturité n'est pas encore atteinte.

### **3. Calcul des pertes**
Les pertes sont calculées comme suit :
\[
\text{Pertes} = (1 - \text{Taux de recouvrement}) \times (\text{Prêts en défaut} \times \text{Montant du prêt})
\]

### **4. Calcul des flux de trésorerie positifs**
Deux types de flux positifs sont considérés :
- **Paiements de principal** : Effectués uniquement à la maturité des prêts qui ne sont pas en défaut.
- **Paiements d'intérêts** : Effectués à chaque période pour les prêts actifs. Les intérêts sont calculés comme :
\[
\text{Intérêts} = \text{Montant du prêt} \times \text{Taux d'intérêt du prêt}
\]

### **5. Flux nets et pricing des tranches**
Les flux nets sont obtenus en soustrayant les pertes aux flux positifs (principal + intérêts). Les flux nets sont ensuite alloués aux différentes tranches (Senior, Mezzanine, Equity) en fonction des limites d'attachement définies.

---

## **Paramètres de la Fonction**

### **Entrées**
- **`correlation_matrix`** *(ndarray)* : Matrice de corrélation entre les prêts du portefeuille.
- **`portfolio`** *(pd.DataFrame)* : Portefeuille de prêts. Doit contenir les colonnes suivantes :
  - `Loan_Amount` : Montant principal du prêt.
  - `Maturity_Years` : Maturité du prêt en années.
  - `Default_Probability` : Probabilité de défaut annuelle.
  - `Interest_Rate` : Taux d'intérêt associé au prêt.
- **`recovery_rate`** *(float, défaut = 0.4)* : Taux de recouvrement en cas de défaut.
- **`tranche_spreads`** *(dict, défaut = `{"Mezzanine": 0.03, "Senior": 0.01}`)* : Spreads des tranches Mezzanine et Senior.
- **`risk_free_rate`** *(float, défaut = 0.03)* : Taux sans risque utilisé pour actualiser les cash flows.
- **`num_samples`** *(int, défaut = 100)* : Nombre de simulations Monte Carlo.
- **`senior_attachment`** *(float, défaut = 0.3)* : Attachement supérieur pour la tranche Senior (en proportion du portefeuille total).
- **`mezz_attachment`** *(float, défaut = 0.1)* : Attachement supérieur pour la tranche Mezzanine (en proportion du portefeuille total).

### **Sorties**
- **`net_cash_flows`** *(ndarray)* : Flux nets (positifs - pertes) pour chaque période et chaque simulation.
- **`tranche_prices`** *(dict)* : Prix moyens des tranches Senior, Mezzanine et Equity.
- **`interest_payments`** *(ndarray)* : Paiements d'intérêts par période pour chaque simulation.
- **`principal_payments`** *(ndarray)* : Paiements du principal par période pour chaque simulation.
- **`loan_states`** *(ndarray)* : États des prêts (0 = actif, 1 = défaut) pour chaque période et chaque simulation.
- **`losses_per_period`** *(ndarray)* : Pertes totales par période pour chaque simulation.
- **`initial_investment`** *(dict)* : Investissement initial pour chaque tranche.
- **`expected_perf`** *(dict)* : Performances attendues des tranches (rendement, perte moyenne, etc.).

---

## **Exemple d'Utilisation**

```python
# Importer la fonction et les librairies nécessaires
import pandas as pd
import numpy as np
from votre_module import pricing_CLO_multi_periode_gaussian

# Créer un portefeuille fictif
portfolio = pd.DataFrame({
    "Loan_Amount": [1_000_000, 2_000_000, 1_500_000],
    "Maturity_Years": [5, 3, 4],
    "Default_Probability": [0.02, 0.03, 0.025],
    "Interest_Rate": [0.05, 0.04, 0.045]
})

# Matrice de corrélation
correlation_matrix = np.array([
    [1.0, 0.2, 0.3],
    [0.2, 1.0, 0.4],
    [0.3, 0.4, 1.0]
])

# Appeler la fonction
results = pricing_CLO_multi_periode_gaussian(
    correlation_matrix,
    portfolio,
    recovery_rate=0.4,
    tranche_spreads={"Mezzanine": 0.03, "Senior": 0.01},
    risk_free_rate=0.03,
    num_samples=1000
)

# Résultats
net_cash_flows, tranche_prices, interest_payments, principal_payments, loan_states, losses_per_period, initial_investment, expected_perf = results

print("Prix des tranches :", tranche_prices)
```

---

## **Notes Importantes**

1. **Hypothèses** :
   - Les probabilités de défaut annuelles sont constantes sur toutes les périodes.
   - La matrice de corrélation reste fixe (peut être modifiée pour refléter des dynamiques temporelles).
   - Les paiements d'intérêts ne sont effectués que si le prêt est actif (pas en défaut).

2. **Extensions possibles** :
   - Incorporer une structure temporelle dans les corrélations.
   - Permettre des probabilités de défaut dynamiques.
   - Ajouter des coûts ou pénalités pour la tranche Equity.

---

Cette fonction offre une grande flexibilité pour analyser les performances et risques liés aux CLOs tout en restant suffisamment rapide pour un usage pratique grâce à l'approche Monte Carlo et l'utilisation des copules. Elle peut être adaptée selon les besoins spécifiques de votre portefeuille ou méthodologie.


