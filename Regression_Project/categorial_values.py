import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

df=pd.read_csv('Ames_my_withouy_missing.csv')

print(df.isnull().sum())
df['MS SubClass']=df['MS SubClass'].apply(str)
my_object_df=df.select_dtypes(include='object')
my_numeric_df=df.select_dtypes(exclude='object')
df_dummies=pd.get_dummies(my_object_df, drop_first=True)
print(df_dummies.head())

final_df=pd.concat([my_numeric_df, df_dummies], axis=1)
final_df.to_csv('Ames_final.csv')