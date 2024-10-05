import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

x = [34, 36, 32, 29, 45, 67, 76, 75, 75, 78, 72, 75, 78, 81, 84, 83, 89, 82, 81, 83]
y = [164, 198, 85, 179, 168, 201, 98, 197, 197, 209, 100, 216, 223, 245, 119, 260, 298, 309, 124, 267]

x_df = pd.DataFrame(x, columns=['Marketing'])

x_df = sm.add_constant(x_df)

results = sm.OLS(y, x_df).fit()
print(results.summary())

fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(results, "Marketing", fig=fig)
plt.show()
