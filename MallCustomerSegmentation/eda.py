import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("data/Mall_Customers.csv")
print(df.head())
print(df.describe())
print(df.info())

# Age distribution
plt.figure(figsize=(10, 4))
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Income distribution
plt.figure(figsize=(10, 4))
sns.histplot(df['Annual Income (k$)'], kde=True)
plt.title('Annual Income Distribution')
plt.show()

# Spending score distribution
plt.figure(figsize=(10, 4))
sns.histplot(df['Spending Score (1-100)'], kde=True)
plt.title('Spending Score Distribution')
plt.show()

# Pairplot of features
sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.show()