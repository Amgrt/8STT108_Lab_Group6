import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel('decathlete2008_dataLabo4.xlsx', sheet_name='Points')
df_scaled = (df - df.mean(numeric_only=True)) / df.std(numeric_only=True)
x = df_scaled[['Run100_pts', 'LJ_pts', 'SP_pts', 'HJ_pts', 'Run400_pts',
               'H_pts', 'DT_pts', 'PV_pts', 'JT_pts', 'Run1500_pts']]
y = df_scaled['Overall']

# Get covariance matrix
# Values and eigenvectors
eig_vals, eig_vecs = np.linalg.eig(x.cov())
# Calculate the total variance
total_var = x.var().sum()
print(f"Total variance: {total_var:.2f}")
# Calculate the sum of the eigenvalues
eig_vals_sum = eig_vals.sum()
print(f"Sum of eigenvalues: {eig_vals_sum:.2f}")
var_exp = eig_vals / total_var
sorted_indices = np.argsort(var_exp)[::-1]
sorted_var_exp = var_exp[sorted_indices]
print(f"Proportion of variance explained by each principal component:\n{sorted_var_exp}")

# Sort the independant variables in order of variance explanation
sorted_index = np.argsort(eig_vals)[::-1]
sorted_eigenvalue = eig_vals[sorted_index]
sorted_eigenvectors = eig_vecs[:, sorted_index]
print(eig_vals)
print(sorted_eigenvalue)
print(sorted_index)

# Visualize the scree plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(sorted_var_exp, marker="o")
ax[0].set_xlabel("Principal component")
ax[0].set_ylabel("Proportion of explained variance")
ax[0].set_title("Scree plot")
ax[1].plot(np.cumsum(sorted_var_exp), marker="o")
ax[1].set_xlabel("Principal component")
ax[1].set_ylabel("Cumulative sum of explained variance")
ax[1].set_title("Cumulative scree plot")
plt.show()

k = 6
principal_components = sorted_eigenvectors[:, :k]
X_new = np.dot(x, principal_components)
print(X_new)
X_new = sm.add_constant(X_new)
results = sm.OLS(y, X_new).fit()
print(results.summary())