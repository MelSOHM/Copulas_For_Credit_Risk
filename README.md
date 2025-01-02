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
