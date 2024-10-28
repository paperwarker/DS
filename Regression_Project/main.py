import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('Ames_final.csv')
X=df.drop('SalePrice', axis=1)
y=df['SalePrice']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import ElasticNet
elastic_model=ElasticNet()

param_grid={'alpha':[0.5, 1, 5, 50, 99], 'l1_ratio':[0.1, 0.3, 0.5, 0.7, 0.9]}

from sklearn.model_selection import GridSearchCV
grid_model=GridSearchCV(estimator=elastic_model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=10)
grid_model.fit(X_train, y_train)
print(grid_model.best_estimator_)

y_pred=grid_model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error
MAE=mean_absolute_error(y_pred, y_test)
RMSE=np.sqrt(mean_squared_error(y_pred, y_test))
print(MAE)
print(RMSE)