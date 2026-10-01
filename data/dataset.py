import pandas as pd
import numpy as np


df = pd.read_csv("IBM-HRAnalytics-Employee-Attrition.csv")

print("First 10 records:", df.head(10))

#print("\nSummary statistics:", df.describe())
print("\nMissing values:", df.isnull().sum())
print("\nData types:", df.dtypes)
