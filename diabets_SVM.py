"""20.11.2004"""
"""diabets.csv faylini SVM model bo'yicha klassifikatsiya qilish va turli funksiyalarda aniqligini hisoblash"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
filename = "diabetes.csv"
data = pd.read_csv(filename)
#print(data.head())
features = data.drop(columns=['Outcome'])
labels = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_model = SVC(kernel='linear', random_state=42)

"""Boshqa kernel funksiyalari: linear(Linear),poly(Polynomial),rbf(RBF),sigmoid(Sigmoid)"""
#svm_model = SVC(kernel='poly', random_state=42)
#svm_model = SVC(kernel='rbf', random_state=42)
#svm_model = SVC(kernel='sigmoid', random_state=42)

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model aniqligi: {:.2f}%".format(accuracy * 100))

