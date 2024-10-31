import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
Data=load_iris()
x=Data.data
y=Data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.2 ,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=MLPClassifier(max_iter=1000)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(y_pred)
print(y_test)
print(accuracy)