import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
data = pd.read_csv('Lab1_Buffalo_Cleaned.csv')

# 2. Descriptive statistics
data_no_total = data[data['Region'].str.lower() != 'total']
female_stats = data_no_total['Female'].describe()
male_stats = data_no_total['Male'].describe()
total_stats = data_no_total['Total'].describe()

print("Female Statistics:\n", female_stats)
print("Male Statistics:\n", male_stats)
print("Total Statistics:\n", total_stats)

# 3. Draw a box plot of females, males, and total population
plt.figure(figsize=(12, 6))

# Box plot of female cattle herd
plt.subplot(1, 3, 1)
plt.boxplot(data_no_total['Female'])
plt.title('Female Cattle')

# Box plot of male cattle herd
plt.subplot(1, 3, 2)
plt.boxplot(data_no_total['Male'])
plt.title('Male Cattle')

# Box plot of total cattle herd
plt.subplot(1, 3, 3)
plt.boxplot(data_no_total['Total'])
plt.title('Total Cattle')

plt.tight_layout()
plt.show()

# 4. Draw a box plot of the total number of each region by region
data.boxplot(column='Total', by='Region', grid=False, figsize=(12, 6))

plt.title('Total Cattle by Region')
plt.xlabel('Region')
plt.ylabel('Total Cattle')
plt.xticks(rotation=45)

plt.show()
