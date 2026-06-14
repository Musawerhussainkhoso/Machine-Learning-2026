import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv(r'D:\ML\Proj02\heart.csv')
df_copy = df.copy()

print(df_copy.head())
#EDA 
print(df_copy.columns)
print(df_copy.shape)
print(df_copy.info())
print(df_copy.describe())
print(df_copy.isnull().sum())
print("Duplicate values in the dataset:")
print(df_copy.duplicated().sum())
#df['HeartDisease'].value_counts()
print(df_copy['HeartDisease'].value_counts())
#plot
print(df_copy['HeartDisease'].value_counts().plot(kind='bar'))
plt.show()
#lets cretae numerical distribution and check the distributions of numericals are correct or not
def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df_copy[var], kde=True)

plt.figure(figsize=(10, 8))

plotting('Age', 1)
plotting('RestingBP', 2)
plotting('Cholesterol', 3)
plotting('MaxHR', 4)

plt.tight_layout()
plt.show()
print(df_copy['Cholesterol'].value_counts())
ch_mean = df.loc[df_copy['Cholesterol'] != 0, 'Cholesterol'].mean()#loc means targeting rows
df_copy['Cholesterol'] = df_copy['Cholesterol'].replace(0, ch_mean)
df_copy['Cholesterol'] = df_copy['Cholesterol'].round(2)
print(df_copy['Cholesterol'].value_counts())
#now for resting bp
restingbp_mean = df.loc[df_copy['RestingBP'] != 0, 'RestingBP'].mean()#loc means targeting rows
df_copy['RestingBP'] = df_copy['RestingBP'].replace(0, restingbp_mean)
df_copy['RestingBP'] = df_copy['RestingBP'].round(2)
print(df_copy['RestingBP'].value_counts())
#again check the distribution of numerical columns after replacing the values
def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df_copy[var], kde=True)

plt.figure(figsize=(10, 8))

plotting('Age', 1)
plotting('RestingBP', 2)
plotting('Cholesterol', 3)
plotting('MaxHR', 4)

plt.tight_layout()
plt.show()
#Categorical values analysis
sns.countplot(x='Sex', data=df_copy, hue='HeartDisease')
plt.show()
sns.countplot(x='ChestPainType', data=df_copy, hue='HeartDisease')
plt.show()
sns.countplot(x='FastingBS', data=df_copy, hue='HeartDisease')
plt.show()
#box plot(can comapre numerical values with categorical values)
sns.boxplot(x='HeartDisease', y='Cholesterol', data=df_copy)
plt.show()
sns.violinplot(x='HeartDisease', y='Age', data=df_copy)
plt.show()
sns.heatmap(df_copy.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
#data preprocessing 
print("Before:", df_copy.shape)
df_encode = pd.get_dummies(df_copy,drop_first=True)
print(df_encode)
df_encode.astype('int')
print(df_encode.dtypes)