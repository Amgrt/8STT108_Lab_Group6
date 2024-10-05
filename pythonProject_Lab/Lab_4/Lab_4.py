import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv('data_Labo4.csv', sep=';')
# Target variable: Whether there are satellites (satellites>0)
df['has_satell'] = (df['satell'] > 0).astype(int)

# 2. Fitting logistic regression model (logit model)
X1 = sm.add_constant(df['width'])
y1 = df['has_satell']
logit_model = sm.Logit(y1, X1)
logit_result = logit_model.fit()
print(logit_result.summary())

# Fit using GLM model
glm_model = sm.GLM(y1, X1, family=sm.families.Binomial())
glm_result = glm_model.fit()
print(glm_result.summary())

# 4. Calculate the log odds of a female crab with a shell width of 25 centimeters
width_25 = np.array([1, 25])
log_odds_25 = logit_result.predict(width_25)
print(f"Log odds for 25 cm shell width: {log_odds_25}")

# 5. Convert log odds into probabilities
probability_25 = np.exp(log_odds_25) / (1 + np.exp(log_odds_25))
print(f"Probability of having satellites for 25 cm shell width: {probability_25}")

# 6. logistic regression model
X2 = sm.add_constant(df['weight'])
logit_model_weight = sm.Logit(y1, X2)
logit_result_weight = logit_model_weight.fit()
print(logit_result_weight.summary())

# 7. Regression equation
beta0, beta1 = logit_result_weight.params
print(f"Regression equation: log(p / (1 - p)) = {beta0} + {beta1} * weight")

# 8. The probability of a female crab weighing 2000 grams having satellites
weight_2000 = np.array([1, 2000])
log_odds_2000 = logit_result_weight.predict(weight_2000)
probability_2000 = np.exp(log_odds_2000) / (1 + np.exp(log_odds_2000))
print(f"Probability of having satellites for 2000g weight: {probability_2000}")
