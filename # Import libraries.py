# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset from seaborn (no need for train.csv)
df = sns.load_dataset('titanic')

# Display first few rows
print(df.head())

# Basic info
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Data Cleaning
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)

# Fix for categorical 'deck' column
df['deck'] = df['deck'].cat.add_categories('Unknown')
df['deck'].fillna('Unknown', inplace=True)

# Convert categorical to numeric
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Verify cleaning
print("\nAfter Cleaning:\n", df.isnull().sum())

# EDA

# Survival count
plt.figure(figsize=(6,4))
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.show()

# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='survived', hue='sex', data=df)
plt.title("Survival by Gender")
plt.show()

# Survival by Class
plt.figure(figsize=(6,4))
sns.countplot(x='survived', hue='pclass', data=df)
plt.title("Survival by Class")
plt.show()

# Age distribution
plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

