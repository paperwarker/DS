import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Advertising.csv')
X=df.drop('sales', axis=1)
y=df['sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from  sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import ElasticNet
base_elastic_net_model=ElasticNet()
param_grid={'alpha':[0.1, 1, 5, 50, 100], 'l1_ratio':[0.1, 0.5, 0.7, 0.95, 0.99, 1]}

from sklearn.model_selection import GridSearchCV
grid_model=GridSearchCV(estimator=base_elastic_net_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2)
grid_model.fit(X_train, y_train)
print(grid_model.best_estimator_)

y_pred=grid_model.predict(X_test)

from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y_pred, y_test)
print(MSE)

