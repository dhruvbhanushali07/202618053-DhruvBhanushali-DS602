# 🍽️ M.Sc. Data Science — Applied Statistical Modeling & Interactive Web Dashboard

**Course:** Statistical Modeling with Python (Lab-4)  
**Target Level:** M.Sc. Data Science (Semester 1)  
**Dataset Selected:** Seaborn Tipping Behavior Dataset (`tips`)  
**Live Application URL:** https://github.com/dhruvbhanushali07/202618053-DhruvBhanushali-DS602

---

## 📌 Project Overview

This project delivers an end-to-end data science application built using Python, `statsmodels`, `scipy.stats`, `plotly`, and `Streamlit`. Moving beyond static notebook environments, this interactive dashboard allows users to perform exploratory data analysis, run automated hypothesis tests (parametric & non-parametric), evaluate Ordinary Least Squares (OLS) regression models, and validate Gauss-Markov assumptions via interactive residual diagnostics.

---

## 📊 Dataset Summary (`tips`)

The dataset contains $N = 244$ dining transaction records captured to measure restaurant tipping behavior.

* **Target Variables:** `tip` (Continuous, $), `tip_pct` (Derived Continuous, %)
* **Predictor Variables:** 
  * `total_bill`: Total cost of the meal ($)
  * `size`: Number of people in the dining party
  * `sex`: Gender of the bill payer (`Male`, `Female`)
  * `smoker`: Smoking section status (`Yes`, `No`)
  * `day`: Day of the week (`Thur`, `Fri`, `Sat`, `Sun`)
  * `time`: Meal timing (`Lunch`, `Dinner`)

---

## 📈 Synthesis of Statistical Findings

### 1. Descriptive & Exploratory Analysis
* **Right Skewness:** Both `total_bill` and `tip` exhibit right-skewed distributions with moderate positive kurtosis. The mean tip is **$3.00** ($\pm \$1.38$) with a median of **$2.90** and an IQR of **$1.66**.
* **Correlation:** A strong positive linear correlation ($r \approx 0.68$) exists between `total_bill` and `tip`. Party `size` also exhibits a positive linear relationship with total tip amount ($r \approx 0.49$).

### 2. Inferential Hypothesis Testing ($\alpha = 0.05$)
* **Normality & Variance Assessment:** Shapiro-Wilk tests on target metrics across groups (e.g., `smoker` vs. `non-smoker`) yield $p < 0.05$, rejecting the assumption of normal distributions.
* **Two-Group Comparison (Smokers vs. Non-Smokers):** Due to non-normality, the non-parametric **Mann-Whitney U Test** was executed. The test yields $p > 0.05$, demonstrating **no statistically significant difference** in median tip amounts between smoking and non-smoking parties.
* **Multi-Group Comparison (ANOVA across Days):** One-Way ANOVA across days of the week reveals no statistically significant difference in mean tip percentages.
* **Categorical Association (Chi-Square Test):** A Chi-Square Test of Independence between `time` (Lunch/Dinner) and `day` confirms significant dependence ($p < 0.001$), reflecting distinct operating schedules.

### 3. OLS Regression & Gauss-Markov Diagnostics

* **Fit Quality:** The multiple linear regression model accounts for approximately **47% of the variance** in tip amounts ($R^2 \approx 0.470$).
* **Key Coefficients:** `total_bill` ($\beta \approx 0.094, p < 0.001$) and party `size` ($\beta \approx 0.19, p < 0.05$) serve as statistically significant positive predictors. Categorical indicators like `sex` and `smoker` do not achieve statistical significance at $\alpha = 0.05$.
* **Multicollinearity:** Variance Inflation Factors (VIF) for all continuous predictors remain well below $5.0$, confirming low levels of multicollinearity.
* **Residual Diagnostics:** Scatter plots of Residuals vs. Fitted values indicate heteroscedasticity increasing slightly at higher total bill amounts, while the Q-Q plot and Jarque-Bera test confirm minor tail-deviation from strict normality.
---

## ⚙️ Installation & Local Setup

### Prerequisites
* Python 3.9+ 
* Git

### Step 1: Clone Repository & Navigate
```bash
git clone [https://github.com/your-username/msc-lab4-statistical-dashboard.git](https://github.com/your-username/msc-lab4-statistical-dashboard.git)
cd msc-lab4-statistical-dashboard
