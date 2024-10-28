import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Указываем среднее значение, среднеквадратическое отклонение и количество сэмплов

def create_ages(mu=50, sigma=13, num_samples=100, seed=42):
    # Указываем значение random seed в той же ячейке, что и вызов метода random -
    # это нужно для того, чтобы получить те же самые данные
    # Мы используем значение 42 (42 это число из комедийного сериала Автостопом по Галактике -
    # Hitchhiker's Guide to the Galaxy)
    np.random.seed(seed)

    sample_ages = np.random.normal(loc=mu, scale=sigma, size=num_samples)
    sample_ages = np.round(sample_ages, decimals=0)

    return sample_ages

sample=create_ages()
#sns.displot(sample, bins=20)
#sns.boxplot(sample)

ser=pd.Series(sample)



q75,q25=np.percentile(sample, [75,25])
IQR=q75-q25



#Ищем у удаляем выбросы
df=pd.read_csv('Ames_Housing_Data.csv')

sns.scatterplot(data=df, x=df['Overall Qual'], y=df['SalePrice'])
plt.show()
sns.scatterplot(data=df, x=df['Gr Liv Area'], y=df['SalePrice'])
plt.show()
drop_index=df[(df['Gr Liv Area']>4000) & (df['SalePrice']<200000)].index
df=df.drop(drop_index, axis=0)
df.to_csv('Ames_my_outliers_removed.csv')