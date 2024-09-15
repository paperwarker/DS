import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("D:\\курс data\\08-Linear-Regression-Models\\Advertising.csv")
df["total_spend"]=df["TV"]+df["radio"]+df["newspaper"]
print(df.head())
#sns.regplot(data=df, x="total_spend", y="sales")


#Простая линейная ригрессия с использованием numpy
X=df["total_spend"]
y=df["sales"]
beta1=np.polyfit(X, y, deg=1)#Вычисляет коэффициенты бэта для указанных X и y в указанной степени
potential_spend=np.linspace(0,500,100)#задаем значение X
predicted_sales=beta1[0]*potential_spend+beta1[1]#вычисляем y
#sns.scatterplot(data=df, x=X, y=y)
#plt.plot(potential_spend, predicted_sales)
#plt.show()

spend=200
predicted_sales=beta1[0]*spend+beta1[1]
print(predicted_sales)


beta2=np.polyfit(X, y, deg=3)
pot_spend=np.linspace(0,500,100)
pred_sales=beta2[0]*pot_spend**3+\
           beta2[1]*pot_spend**2+\
           beta2[2]*pot_spend+\
           beta2[3]
#sns.scatterplot(data=df, x=X, y=y)
#plt.plot(pot_spend, pred_sales, color="red")


#fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,6))

'''
axes[0].plot(df['TV'],df['sales'],'o')
axes[0].set_ylabel("Sales")
axes[0].set_title("TV Spend")

axes[1].plot(df['radio'],df['sales'],'o')
axes[1].set_title("Radio Spend")
axes[1].set_ylabel("Sales")

axes[2].plot(df['newspaper'],df['sales'],'o')
axes[2].set_title("Newspaper Spend");
axes[2].set_ylabel("Sales")
plt.tight_layout();
#plt.show()
'''

#Линейная регрессия в SKlearn

#Разбиение на признаки и целевую переменную
X=df.drop('sales', axis=1)#Признаки
y=df['sales']#целевая переменная

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)#Разбивает данные на обучающий и тестовый наборы

from sklearn.linear_model import LinearRegression#импорт линейной регрессии
model=LinearRegression()
model.fit(X_train, y_train)
test_prediction=model.predict(X_test)

#Выявление ошибки
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_sales=df['sales'].mean()
print(mean_absolute_error(y_test, test_prediction))
print(mean_squared_error(y_test, test_prediction))
print(np.sqrt(mean_squared_error(y_test, test_prediction)))

#Анализ остатков
test_residuals=y_test-test_prediction
sns.scatterplot(x=y_test, y=test_residuals)
plt.axhline(y=0, color="red")
sns.displot(test_residuals, bins=25, kde=True)
#plt.show()

#Создание финальной модели

final_model=LinearRegression()
X=X.drop("total_spend", axis=1)
final_model.fit(X, y)
print(final_model.coef_)
print(X.head())

y_hat=final_model.predict(X)

fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,6))

axes[0].plot(df['TV'],df['sales'],'o')
axes[0].plot(df['TV'],y_hat,'o',color='red')
axes[0].set_ylabel("Sales")
axes[0].set_title("TV Spend")

axes[1].plot(df['radio'],df['sales'],'o')
axes[1].plot(df['radio'],y_hat,'o',color='red')
axes[1].set_title("Radio Spend")
axes[1].set_ylabel("Sales")

axes[2].plot(df['newspaper'],df['sales'],'o')
axes[2].plot(df['newspaper'],y_hat,'o',color='red')
axes[2].set_title("Newspaper Spend");
axes[2].set_ylabel("Sales")
plt.tight_layout();
plt.show()

#Сохранение модели
from joblib import dump, load
dump(final_model, "final_sales_model.joblib")




