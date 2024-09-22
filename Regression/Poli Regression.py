import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Advertising.csv')

X=df.drop('sales', axis=1)
y=df['sales']

from sklearn.preprocessing import PolynomialFeatures
#Создание полиноимиальных признаков
polynomial_converter=PolynomialFeatures(degree=2, include_bias=False)
polynomial_converter.fit(X)
poly_features=polynomial_converter.transform(X)

from sklearn.model_selection import train_test_split
#Разбиваем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression
#Создаем регрессию

model=LinearRegression()
model.fit(X_train, y_train)
test_prediction=model.predict(X_test)


from sklearn.metrics import mean_absolute_error, mean_squared_error
#Оцениваем ошибки

MAE=mean_absolute_error(y_test, test_prediction)
MSE=mean_squared_error(y_test, test_prediction)
RMSE=np.sqrt(MSE)


#Поиск степени для полиномиальной регресси и вычесление переобученности или недообученности
train_rmse_errors=[]
test_rmse_errors=[]

for i in range (1,10):
    polynomial_converter=PolynomialFeatures(degree=i, include_bias=False)
    poly_features=polynomial_converter.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)

    model=LinearRegression()
    model.fit(X_train, y_train)
    test_prediction=model.predict(X_test)
    train_prediction=model.predict(X_train)

    train_rmse=np.sqrt(mean_squared_error(y_train, train_prediction))
    test_rmse=np.sqrt(mean_squared_error(y_test, test_prediction))

    train_rmse_errors.append(train_rmse)
    test_rmse_errors.append(test_rmse)

plt.plot(range(1,10), train_rmse_errors)
plt.plot(range(1,10), test_rmse_errors)



final_poly_converter=PolynomialFeatures(degree=3, include_bias=False)
final_poly_features=final_poly_converter.fit_transform(X)
final_model=LinearRegression()
final_model.fit(final_poly_features, y)

from joblib import dump, load

dump(final_model, 'final_poly_model.joblib')
dump(final_poly_converter, 'final_poly_converter.joblib')
