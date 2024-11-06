"""06.11.2024 </> Abdulahad </>"""
"""Sun'iy intellekt fanidan """

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data={'area':[1200,1500,1700,2000,2300,2500,2700,3000,3300,3600],
      'rooms':[3,3,4,4,5,5,5,6,6,7],
      'kitchens':[1,1,1,1,2,2,2,2,3,3],
      'bathrooms':[1,2,2,2,3,3,3,4,4,5],
      'price':[30000,35000,40000,45000,50000,55000,60000,65000,70000,75000]}
df=pd.DataFrame(data)
x=df[['area','rooms','kitchens','bathrooms']].values
y=df['price'].values
scaler=StandardScaler()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.2 ,random_state=42)
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
MSE=mean_squared_error(y_pred,y_test)
print("Bashorat qilingan qiymat: ",y_pred)
print("Haqiqiy qiymat: ",y_test)
print("O'rtacha kvadratik xatolik: ",MSE)