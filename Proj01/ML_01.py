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
