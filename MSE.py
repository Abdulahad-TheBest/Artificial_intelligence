"""
25.11.2024
SVM model bo'yicha Regressiya masalasini yechish
MSE ni hisoblash va Haqiqiy va Bashorat qilingan qiymatlarni chop etish
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
data = pd.read_csv("Giper147.csv") # Giper147.csv data set orqali tuzilgan
X = data.drop(columns=['Buyi'])
y = data['Buyi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svr_model = SVR(kernel='linear')
svr_model.fit(X_train, y_train)
y_pred = svr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nMSE:", mse)
y_test = np.array(y_test)
print("\nHaqiqiy va bashorat qilingan qiymatlar:")
for i in range(min(10, len(y_test))):
    print(f"Haqiqiy qiymat: {y_test[i]}, Bashorat qilingan qiymat: {y_pred[i]}")
