import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open ('Ames_Housing_Feature_Description.txt', 'r') as f:
    print(f.read())

df=pd.read_csv('Ames_my_outliers_removed.csv')

#Столбцы с id можно удалять для моделей МО, так как pandas назначает свой id
df=df.drop('PID', axis=1)

#Создаем функция для поиска процента пустых строк
def pecent_missing(my_df):
    result=100*df.isnull().sum()/len(my_df)
    result=result[result>0].sort_values()
    return result

percent_nan=pecent_missing(df)
sns.barplot(x=percent_nan.index, y=percent_nan)
plt.xticks(rotation=90)#Поворачиваем названия индексов на 90 градусов
plt.ylim(0,1)


#Заполнение данных по строкам
#print(percent_nan[percent_nan<1])#Ищем признаки с процентом пустых строк меньше единицы
#print(df[df['Electrical'].isnull()]['Garage Area'])
#print(df[df['Bsmt Half Bath'].isnull()])

df=df.dropna(axis=0, subset=['Electrical', 'Garage Area'])#Удаляем строки с пустыми значениями
percent_nan=pecent_missing(df)
sns.barplot(x=percent_nan.index, y=percent_nan)
plt.xticks(rotation=90)#Поворачиваем названия индексов на 90 градусов
plt.ylim(0,1)
plt.show()

#print(df[df['Bsmt Half Bath'].isnull()])
#print(df[df['Bsmt Half Bath'].isnull()])
#print(df[df['Bsmt Unf SF'].isnull()])

#Операция для числовых колонок-запишем нули
bsmt_num_col=['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF','Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath']
df[bsmt_num_col]=df[bsmt_num_col].fillna(0)#Заменяем отсутствующие значения на нули

#Операция для текстовых колонок
bsmt_str_col=['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
df[bsmt_str_col]=df[bsmt_str_col].fillna('None')

percent_nan=pecent_missing(df)
sns.barplot(x=percent_nan.index, y=percent_nan)
plt.xticks(rotation=90)#Поворачиваем названия индексов на 90 градусов
plt.ylim(0,1)
plt.show()

df['Mas Vnr Type']=df['Mas Vnr Type'].fillna('None')
df['Mas Vnr Area']=df['Mas Vnr Area'].fillna(0)

percent_nan=pecent_missing(df)
sns.barplot(x=percent_nan.index, y=percent_nan)
plt.xticks(rotation=90)#Поворачиваем названия индексов на 90 градусов
plt.show()

#Работа со строками
garage_str_cols=['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[garage_str_cols]=df[garage_str_cols].fillna('None')


df['Garage Yr Blt']=df['Garage Yr Blt'].fillna(0)

df=df.drop(['Alley', 'Pool QC', 'Misc Feature', 'Fence'], axis=1)
percent_nan=pecent_missing(df)
sns.barplot(x=percent_nan.index, y=percent_nan)
plt.xticks(rotation=90)#Поворачиваем названия индексов на 90 градусов
plt.show()

df['Fireplace Qu']=df['Fireplace Qu'].fillna('None')
df['Lot Frontage']=df.groupby('Neighborhood')['Lot Frontage'].transform(lambda value: value.fillna(value.mean()))#Заменяем отсутствующие значения в Lot Frontage средним значением
df['Lot Frontage']=df['Lot Frontage'].fillna(0)
df.to_csv('Ames_my_withouy_missing.csv')