import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Advertising.csv')

X=df.drop('sales', axis=1)
y=df['sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import Ridge
ridge_model=Ridge(alpha=100)
ridge_model.fit(X_train, y_train)
y_pred=ridge_model.predict(X_test)

from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y_test, y_pred)
print(MSE)

ridge_model_two=Ridge(alpha=1)
ridge_model_two.fit(X_train, y_train)
y_pred_two=ridge_model_two.predict(X_test)
MSE=mean_squared_error(y_pred_two, y_test)
print(MSE)