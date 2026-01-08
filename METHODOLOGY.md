# Methodology: Counterfactual Demand Estimation

This document provides technical details on the causal inference approach used in this project.

## Table of Contents
1. [The Causal Problem](#the-causal-problem)
2. [Instrumental Variables Approach](#instrumental-variables-approach)
3. [Two-Stage Least Squares Implementation](#two-stage-least-squares-implementation)
4. [Heterogeneous Treatment Effects](#heterogeneous-treatment-effects)
5. [Validation Strategy](#validation-strategy)
6. [Assumptions and Limitations](#assumptions-and-limitations)

---

## The Causal Problem

### Potential Outcomes Framework

For each campaign $i$, we observe:
- $Y_i$: Funding ratio (pledged / goal)
- $D_i$: Treatment (average reward price)
- $X_i$: Covariates (category, goal ambition, duration, etc.)

We want to estimate the **causal effect** of price on funding:
$$\tau = E[Y_i(d') - Y_i(d)]$$

where $Y_i(d)$ is the potential outcome under price $d$.

### The Endogeneity Problem

Simple OLS regression gives:
$$Y_i = \alpha + \beta D_i + X_i'\gamma + \epsilon_i$$

But $\beta_{OLS}$ is **biased** because:
$$E[\epsilon_i | D_i] \neq 0$$

**Why?** Unobserved campaign quality $U_i$ affects both:
- Pricing decisions (high-quality → higher prices)
- Outcomes (high-quality → better funding)

This creates **omitted variable bias**:
$$\beta_{OLS} = \beta_{causal} + \underbrace{\text{Cov}(D_i, U_i) / \text{Var}(D_i)}_{\text{bias from confounding}}$$

---

## Instrumental Variables Approach

### Instrument Selection

We use **launch timing** as instruments:
1. `is_weekend_launch`: Binary indicator for Saturday/Sunday launches
2. `holiday_proximity`: Binary indicator for launches within 7 days of US holidays
3. `trend_spike`: Binary indicator for above-median category trend

### Validity Conditions

For valid instruments, we need:

1. **Relevance**: $\text{Cov}(Z_i, D_i) \neq 0$
   - Launch timing must affect visibility/traffic patterns

2. **Exclusion Restriction**: $\text{Cov}(Z_i, \epsilon_i) = 0$
   - Launch timing must not directly affect funding except through price

3. **Independence**: $Z_i \perp U_i$
   - Launch timing must be independent of unobserved quality

### Justification

| Condition    | Argument                                                     |
| ------------ | ------------------------------------------------------------ |
| Relevance    | Weekend launches have different visibility patterns          |
| Exclusion    | Launch day doesn't directly determine funding success        |
| Independence | Creators choose launch dates based on readiness, not quality |

**Caveat**: If creators strategically time launches based on unobserved factors, exclusion restriction may be violated.

---

## Two-Stage Least Squares Implementation

### Stage 1: Price Prediction

$$D_i = \pi_0 + \pi_1 Z_i + X_i'\pi_2 + \nu_i$$

We regress price on instruments and controls to get predicted prices $\hat{D}_i$.

### Stage 2: Outcome Regression

$$Y_i = \alpha + \beta \hat{D}_i + X_i'\gamma + \eta_i$$

Using predicted prices removes the correlation with unobserved confounders.

### Diagnostics

| Diagnostic              | Purpose              | Threshold                |
| ----------------------- | -------------------- | ------------------------ |
| First-stage F-statistic | Instrument strength  | F > 10                   |
| Sargan-Hansen J-test    | Overidentification   | p > 0.05                 |
| Durbin-Wu-Hausman       | OLS vs IV difference | p < 0.05 means IV needed |

### Code Implementation

```python
from linearmodels.iv import IV2SLS

# Define variables
dependent = df['funding_ratio']
exog = df[['const', 'goal_ambition', 'duration', 'trend_index']]
endog = df[['avg_reward_price']]
instruments = df[['is_weekend_launch', 'holiday_proximity', 'trend_spike']]

# Fit model
model = IV2SLS(dependent, exog, endog, instruments).fit(cov_type='robust')
```

---

## Heterogeneous Treatment Effects

### Motivation

The average treatment effect (ATE) may hide important heterogeneity:
- Tech products vs art projects
- Low-goal vs high-goal campaigns
- Novice vs experienced creators

### Causal Forest

We use the **Causal Forest** algorithm (Athey & Imbens, 2016):

1. Split data into estimation and splitting samples
2. For each split, estimate treatment effects
3. Recursively partition to find subgroups with different effects
4. Average across trees for stable estimates

### Implementation

```python
from econml.dml import CausalForestDML

cf_model = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100),
    model_t=RandomForestRegressor(n_estimators=100),
    discrete_treatment=False
)

cf_model.fit(Y, T, X=moderators)
treatment_effects = cf_model.effect(X)
```

### Interpretation

Treatment effects can be summarized by:
- Distribution (histogram)
- Conditional means (by category, goal quartile)
- Best linear projection on moderators

---

## Validation Strategy

### 1. Temporal Validation

**Question**: Can we predict outcomes for campaigns that changed strategy?

**Method**:
1. Identify campaigns with high price variation across reward tiers
2. Treat variation as proxy for "experimentation"
3. Predict counterfactual outcomes
4. Measure prediction error

### 2. Placebo Tests

**Question**: Does the model detect changes where none exist?

**Method**:
1. Select campaigns with low price variation
2. Randomly perturb features by 1%
3. Measure "phantom effect"
4. Should be ~0 if model is well-calibrated

### 3. Manski Bounds

**Question**: Are estimates within theoretical limits?

**Method**:
1. Compute worst-case bounds assuming unobserved confounding
2. Check if point estimates fall within bounds
3. Wide bounds suggest high uncertainty

### 4. Temporal Cross-Validation

**Question**: Are causal effects stable over time?

**Method**:
1. Train on 2020-2022 data
2. Test on 2023-2024 data
3. Compare coefficients across periods
4. Small changes suggest stable relationships

---

## Assumptions and Limitations

### Key Assumptions

| Assumption              | Implication if Violated                               |
| ----------------------- | ----------------------------------------------------- |
| SUTVA (no interference) | If campaigns compete, effects are biased              |
| Exclusion restriction   | If launch timing affects quality, IV is invalid       |
| Monotonicity            | If effects have different signs, LATE is unclear      |
| Overlap                 | If some prices never observed, extrapolation is risky |

### Known Limitations

1. **Synthetic Data**: This demo uses realistic synthetic data due to Kickstarter blocking. Real data would improve estimates.

2. **Weak Instruments**: Launch timing has limited correlation with pricing (F-stat may be low).

3. **Local Effects**: 2SLS estimates Local Average Treatment Effect (LATE), not ATE.

4. **Linearity**: Model assumes linear demand curves; real curves may be kinked.

---

## References

1. Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press.

2. Athey, S., & Imbens, G. W. (2016). Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353-7360.

3. Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.

4. Manski, C. F. (1990). Nonparametric bounds on treatment effects. *American Economic Review*, 80(2), 319-323.

5. Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV regression. *Identification and Inference for Econometric Models*, 80-108.
