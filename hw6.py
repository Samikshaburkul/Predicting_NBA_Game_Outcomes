#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rfit

# Part I
titanic = rfit.dfapi('Titanic', 'id')
titanic.dropna(inplace=True)
print(titanic)

# Part II
nfl = rfit.dfapi('nfl2008_fga')
nfl.dropna(inplace=True)
print(nfl)

import statsmodels.api as sm

#Define your features and target variable
titanic['age'] = pd.to_numeric(titanic['age'], errors='coerce')
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

X = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch']]
X = pd.get_dummies(X, columns=['sex'], drop_first=True)  # Convert categorical to numerical
y = titanic['survived']

# Add a constant to the features
X = sm.add_constant(X)

# Fit the logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Display the summary
print(result.summary())

# %%
