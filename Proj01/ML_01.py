import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.filterwarnings('ignore')
data = pd.read_csv(r'D:\ML\Proj01\insurance.csv')
df = data.copy()
#print(data.head())
print(np.__version__)
#EDA
print(df.shape)
df.info()
print(df.head())
print(df.tail())
print(df.describe()) 
print(df.isnull().sum()) 
#Visualization
#first check columnns 
print(df.columns)
num_col = [
    'age',
    'bmi',
    'children',
    'charges'
]# checking distributions


for col in num_col:
    plt.figure(figsize=(6,4))# 6 to 4 will be pixel size of the graph
    sns.histplot(df[col], kde=True)#histogram use for continuous data and kde is for density plot
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
sns.countplot(x= 'children', data=df)#countplot is used for categorical data and x is the column name and data is the dataframe name
plt.title('Distribution of Children')
plt.xlabel('Number of Children')
plt.ylabel('Count')
plt.show()
sns.countplot(x = 'sex', data=df)
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
sns.countplot(x = 'smoker', data=df)
plt.title('Distribution of Smoker')
plt.xlabel('Smoker')
plt.ylabel('Count')
plt.show()

#now i will connect input and output variables then see the analysis, like what distribution is going on.
for col in num_col:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

#co-relation
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')

plt.show()
#Data Cleaning and Preprocessing
print("Before:", df.shape)
print("Duplicates:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("After:", df.shape)
df.isnull().sum()
print(df.dtypes)
print(df['sex'].value_counts())
print(df['smoker'].value_counts())
#label encoding 
df['sex']=df['sex'].map({'male': 0, 'female': 1})
print(df.head())
df['smoker']=df['smoker'].map({'no': 0, 'yes': 1})
print(df.head())
#rename some columns 
df.rename(columns={
    'sex': 'is_female',
    'smoker': 'is_smoker'
}, inplace=True)
print(df.head())
#hot encoding
df = pd.get_dummies(df, columns=['region'], drop_first=True)
print(df.head())
df=df.astype(int)
print(df)



