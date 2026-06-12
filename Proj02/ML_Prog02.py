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