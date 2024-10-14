import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Advertising.csv')

X=df.drop('sales', axis=1)
y=df['sales']

from sklearn.preprocessing import PolynomialFeatures
polynomial_converter=PolynomialFeatures(degree=3, include_bias=False)
polynomial_features=polynomial_converter.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(polynomial_features, y, test_size=0.3, random_state=101)

#Масштабирование признаков
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#Ridge регрессия

from sklearn.linear_model import Ridge
ridge_model=Ridge(alpha=10)
ridge_model.fit(X_train, y_train)
test_prediction=ridge_model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
MAE=mean_absolute_error(test_prediction, y_test)
RMSE=np.sqrt(mean_squared_error(test_prediction, y_test))


from sklearn.linear_model import RidgeCV
ridge_cv_model=RidgeCV(alphas=(0.1, 1.0, 10.0), scoring='neg_mean_absolute_error')
ridge_cv_model.fit(X_train, y_train)
print(ridge_cv_model.alpha_)

test_prediction=ridge_cv_model.predict(X_test)
MAE=mean_absolute_error(test_prediction, y_test)
RMSE=np.sqrt(mean_squared_error(test_prediction, y_test))
print(MAE)
print(RMSE)


#Lasso регрессия (least absolute shrinkage and selection operator)

from sklearn.linear_model import LassoCV

lasso_cv_model=LassoCV(eps=0.1, n_alphas=100, cv=5)
lasso_cv_model.fit(X_train, y_train)
print(lasso_cv_model.alpha_)

test_prediction_lasso=lasso_cv_model.predict(X_test)
MAE=mean_absolute_error(test_prediction_lasso, y_test)
RMSE=np.sqrt(mean_squared_error(test_prediction_lasso, y_test))
print(MAE)
print(RMSE)

#Elastic Net

from sklearn.linear_model import ElasticNetCV

elasticNet_cv_model=ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], eps=0.001, n_alphas=100, max_iter=1000000)
elasticNet_cv_model.fit(X_train, y_train)
elastic_predict=elasticNet_cv_model.predict(X_test)

MAE=mean_absolute_error(elastic_predict, y_test)
RMSE=np.sqrt(mean_squared_error(elastic_predict, y_test))
print(MAE)
print(RMSE)

