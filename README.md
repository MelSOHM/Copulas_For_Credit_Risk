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

- **Pricing Framework**: Develop a pricing mechanism for the CLO tranches based on expected losses, investor risk preferences, and market conditions (e.g., interest rates, recovery rates). Monte-Carlo pricing using the copulas ? ✅

- **Risk Analysis**: Evaluate the tail risk and sensitivity of each tranche under different economic scenarios using stress testing and scenario analysis.

- **Comparison Metrics**: Define clear metrics (e.g., VaR, expected shortfall, tranche loss distributions) to compare the performance of Gaussian and Archimedean copulas.

- **Visualization**: Create visualizations to represent default correlations, tranche loss distributions, and risk profiles for different copula models. 2D plot of samples drawn from the 2 different copulas.

- **Documentation and Automation**: Develop a detailed user guide for the pricing tool, including examples of how to use it. Automate repetitive tasks for efficiency.

- **Insights and Recommendations**: Summarize findings on the suitability of each copula model for pricing CLOs and provide recommendations for practical use cases in infrastructure loans.

- **Analyse the results and test on insightful data**

- **Write the report**

# Documentation:

## CLO Pricing Function Using Gaussian Copulas

This function implements a multi-period pricing model for Collateralized Loan Obligations (CLOs), using Gaussian copulas to simulate default correlations between loans in a portfolio. It calculates losses, positive and negative cash flows, and estimates the prices of tranches (Senior, Mezzanine, and Equity) across multiple periods.

---

## **Methodology Description**

### **1. Default Simulation Using a Gaussian Copula**
The function starts by simulating loan defaults in the portfolio using a Gaussian copula to account for dependencies between defaults.

- **Initial Step**: Annual default probabilities of loans are used for the first period.
- **Subsequent Steps**: Active loans (those not in default and within their maturity period) are identified. Correlated samples are generated for these loans at each subsequent period.
- **Conditionality**: Defaults for a given period are simulated only for active loans, considering conditional correlations defined by the copula.

### **2. Identification of Active Loans**
A loan is considered active if:
- It has not defaulted in any previous periods.
- Its maturity period has not yet been reached.

### **3. Loss Calculation**
Losses are calculated as:
$\[
\text{Losses} = (1 - \text{Recovery Rate}) \times (\text{Defaulted Loans} \times \text{Loan Amount})
\] $

### **4. Positive Cash Flow Calculation**
Two types of positive cash flows are considered:
- **Principal Payments**: Paid only at loan maturity if the loan is not in default.
- **Interest Payments**: Paid at each period for active loans. Interest is calculated as:
  
$ \[
\text{Interest} = \text{Loan Amount} \times \text{Loan Interest Rate}
\] $

### **5. Net Cash Flows and Tranche Pricing**
Net cash flows are obtained by subtracting losses from positive cash flows (principal + interest). Net cash flows are then allocated to different tranches (Senior, Mezzanine, Equity) based on the defined attachment points.

---

## **Function Parameters**

### **Inputs**
- **`correlation_matrix`** *(ndarray)*: Correlation matrix between loans in the portfolio.
- **`portfolio`** *(pd.DataFrame)*: Loan portfolio. Must include the following columns:
  - `Loan_Amount`: Loan principal amount.
  - `Maturity_Years`: Loan maturity in years.
  - `Default_Probability`: Annual default probability.
  - `Interest_Rate`: Interest rate associated with the loan.
- **`recovery_rate`** *(float, default = 0.4)*: Recovery rate in case of default.
- **`tranche_spreads`** *(dict, default = `{"Mezzanine": 0.03, "Senior": 0.01}`)*: Spreads for Mezzanine and Senior tranches.
- **`risk_free_rate`** *(float, default = 0.03)*: Risk-free rate used to discount cash flows.
- **`num_samples`** *(int, default = 100)*: Number of Monte Carlo simulations.
- **`senior_attachment`** *(float, default = 0.3)*: Senior tranche attachment point (as a proportion of total portfolio).
- **`mezz_attachment`** *(float, default = 0.1)*: Mezzanine tranche attachment point (as a proportion of total portfolio).

### **Outputs**
- **`net_cash_flows`** *(ndarray)*: Net cash flows (positive flows - losses) for each period and simulation.
- **`tranche_prices`** *(dict)*: Average prices of Senior, Mezzanine, and Equity tranches.
- **`interest_payments`** *(ndarray)*: Interest payments per period for each simulation.
- **`principal_payments`** *(ndarray)*: Principal payments per period for each simulation.
- **`loan_states`** *(ndarray)*: Loan states (0 = active, 1 = default) for each period and simulation.
- **`losses_per_period`** *(ndarray)*: Total losses per period for each simulation.
- **`initial_investment`** *(dict)*: Initial investment for each tranche.
- **`expected_perf`** *(dict)*: Expected performance of tranches (yield, average loss, etc.).
Remark: the format of 3D arrays are : (simulation ID, Loan, Time)
---

## **Example Usage**

```python
# Import the function and necessary libraries
import pandas as pd
import numpy as np
from your_module import pricing_CLO_multi_periode_gaussian

# Create a sample portfolio
portfolio = pd.DataFrame({
    "Loan_Amount": [1_000_000, 2_000_000, 1_500_000],
    "Maturity_Years": [5, 3, 4],
    "Default_Probability": [0.02, 0.03, 0.025],
    "Interest_Rate": [0.05, 0.04, 0.045]
})

# Correlation matrix
correlation_matrix = np.array([
    [1.0, 0.2, 0.3],
    [0.2, 1.0, 0.4],
    [0.3, 0.4, 1.0]
])

# Call the function
results = pricing_CLO_multi_periode_gaussian(
    correlation_matrix,
    portfolio,
    recovery_rate=0.4,
    tranche_spreads={"Mezzanine": 0.03, "Senior": 0.01},
    risk_free_rate=0.03,
    num_samples=1000
)

# Results
net_cash_flows, tranche_prices, interest_payments, principal_payments, loan_states, losses_per_period, initial_investment, expected_perf = results

print("Tranche prices:", tranche_prices)
```

---

## **Key Notes**

1. **Assumptions**:
   - Annual default probabilities are constant across all periods.
   - The correlation matrix remains fixed (can be adapted to reflect temporal dynamics).
   - Interest payments are made only if the loan is active (not in default).

2. **Potential Extensions**:
   - Incorporate a temporal structure in correlations.
   - Allow dynamic default probabilities.
   - Add costs or penalties for the Equity tranche.
   - Incorporate loan pre-payement



