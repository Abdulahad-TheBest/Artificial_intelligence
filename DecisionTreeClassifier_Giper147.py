
"""14.11.2024"""
"""Giper147 bazasini Qarorlar daraxti orqali vizuall ko'rinishi"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv('Giper147.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
plot_tree(model, filled=True)
plt.show()
