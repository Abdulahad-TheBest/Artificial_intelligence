# 06.11.2024
# Uy narxlarini bashorat qiluvchi dastur

""" </> Abdulahad-TheBest </> """

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Ma'lumotlarni yuklash
data = pd.read_csv('house_price_regression_dataset.csv')

# X va y ni ajratib olish
X = data.drop(columns=['House_Price'])
y = data['House_Price']

# Ma'lumotlarni train va test to'plamlarga bo'lish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standartlashtirish
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelni yaratish va o'rgatish
model = MLPRegressor(max_iter=1000, tol=1e-4, random_state=42)  # max_iterni oshirdik va tolni kamaytirdik
model.fit(X_train, y_train)

# Bashorat qilish
y_pred = model.predict(X_test)

# Natijalarni chop etish
for i in range(len(y_test)):
    print(y_test.iloc[i], y_pred[i])
