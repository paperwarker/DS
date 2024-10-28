import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
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

from sklearn.model_selection import cross_val_score
scores=cross_val_score(ridge_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
metric_mean=abs(scores.mean())

ridge_model=Ridge(alpha=1)
scores=cross_val_score(ridge_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
metric_mean=abs(scores.mean())

ridge_model.fit(X_train, y_train)
y_final=ridge_model.predict(X_test)

from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y_test, y_final)
print(MSE)