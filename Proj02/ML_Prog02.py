import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv(r'D:\ML\Proj02\heart.csv')
df_copy = df.copy()

print(df_copy.head())