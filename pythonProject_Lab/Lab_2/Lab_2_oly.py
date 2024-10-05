import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_excel('olympics.xlsx', header=None)

# Remove the first line
df = df.drop(index=[0, 0])

# Set new column name
columns = ['Country', 'Summer', 'Summer_Gold', 'Summer_Silver', 'Summer_Bronze', 'Summer_Total',
           'Winter', 'Winter_Gold', 'Winter_Silver', 'Winter_Bronze','Winter_Total',
           'Games', 'Games_Gold', 'Games_Silver', 'Games_Bronze', 'Combined_total']
df.columns = columns

# Supplement missing values
df[df.select_dtypes(include='number').columns] = df.select_dtypes(include='number').fillna(0)

# Type conversion
numeric_cols = df.columns[1:]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# Remove duplicate values
df = df.drop_duplicates(subset=['Country'])

# Ensure that the value of Total is correct
df['Summer_Total'] = df[['Summer_Gold', 'Summer_Silver', 'Summer_Bronze']].sum(axis=1)
df['Winter_Total'] = df[['Winter_Gold', 'Winter_Silver', 'Winter_Bronze']].sum(axis=1)
df['Combined_total'] = df[['Games_Gold', 'Games_Silver', 'Games_Bronze']].sum(axis=1)
if 'Totals' in df['Country'].values and len(df) > 1:
    totals = df.iloc[:-1, 1:].sum()  # Skip the last row ('Total 'row) and the first column ('country' column)
    df.iloc[-1, 1:] = totals.values

# View
print(df)

# Save
# df.to_excel('cleaned_olympics.xlsx', index=False)

df = pd.read_excel('cleaned_olympics.xlsx')
df = df[df['Country'] != 'Totals']

# Draw the top 10 countries in total summer medals
plt.figure(figsize=(10, 6))
sns.barplot(x='Summer_Total', y='Country', data=df.nlargest(10, 'Summer_Total'))
plt.title('Top 10 Countries by Summer Total Medals')
plt.xlabel('Total Medals')
plt.ylabel('')
plt.xticks(rotation=45)  # Rotate x-axis
plt.tight_layout()
plt.show()

# Draw the top 10 countries in total winter medals
plt.figure(figsize=(10, 6))
sns.barplot(x='Winter_Total', y='Country', data=df.nlargest(10, 'Winter_Total'))
plt.title('Top 10 Countries by Winter Total Medals')
plt.xlabel('Total Medals')
plt.ylabel('')
plt.xticks(rotation=45)  # Rotate x-axis
plt.tight_layout()
plt.show()

# Draw the top 10 countries in total medal count (summer+winter)
combined_medals = df['Summer_Total'] + df['Winter_Total']
df['Combined_Medals'] = combined_medals

plt.figure(figsize=(10, 6))
sns.barplot(x='Combined_Medals', y='Country',
            data=df.nlargest(10, 'Combined_Medals').sort_values(by='Combined_Medals', ascending=False))
plt.title('Top 10 Countries by Combined Total Medals')
plt.xlabel('Total Medals (Summer + Winter)')
plt.ylabel('')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()