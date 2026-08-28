# A/B Test Analysis Report: Checkout Flow Redesign

## Overview
We conducted an A/B test to evaluate a new checkout flow (Treatment) against the current version (Control). The experiment involved 2,000 users (1,000 per group) and focused on two metrics: session duration (continuous) and conversion rate (binary).

## Key Findings

### 1. Session Duration
- **Control Mean:** 7.9133 min
- **Treatment Mean:** 8.2559 min
- **Statistical Test (Welch t-test):** p-value = 0.0109 (Significant)
- **Effect Size (Cohen's d):** 0.1140
- **95% Confidence Interval:** [0.0791, 0.6060]

The difference in session duration is statistically significant (p < 0.05), and the confidence interval does not include zero. However, the effect size (Cohen's d ≈ 0.11) is considered **small**, indicating that while the change is real, its practical impact on user behavior is limited.

### 2. Conversion Rate
- **Control Rate:** 10.8%
- **Treatment Rate:** 13.4%
- **Statistical Test (Chi-squared):** p-value = 0.0865 (Not Significant)
- **95% Confidence Interval:** [-0.0026, 0.0546]

The increase in conversion rate (2.6%) is **not statistically significant** at the alpha = 0.05 level. The p-value (0.0865) is above the threshold, and the 95% confidence interval includes zero, meaning we cannot rule out that this difference occurred by chance.

## Recommendation
Based on the evidence, we **do not recommend** a full rollout of the new checkout flow at this time. While there is a statistically significant increase in session duration, the effect size is very small, and more importantly, the primary goal—increasing the conversion rate—was not achieved with statistical significance. We suggest further iteration on the design or increasing the sample size to clarify the potential impact on conversions.

## How to run
venv
pip install -r requirements.txt
python -m pytest test_abtest.py -v
python3 abtest.py