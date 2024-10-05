import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_excel('BostonHousing.xlsx')
print(df.head())

# Check for missing values in the data
print(df.isnull().sum())

# Delete missing values
df_cleaned = df.dropna()

# Draw paired scatter plots
selected_columns = ['crim', 'rm', 'tax', 'medv']
sns.pairplot(df_cleaned[selected_columns])
plt.suptitle('Pairwise Scatter Plots of Boston Housing', y=1.02)
plt.show()

# Draw a correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Boston Housing')
plt.show()
