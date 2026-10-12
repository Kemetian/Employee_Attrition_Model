import pandas as pd
import numpy as np


hr_analytics_df = pd.read_csv("IBM-HRAnalytics-Employee-Attrition.csv")

print("First 5 records:", hr_analytics_df.head(5))

#print("\nSummary statistics:", hr_analytics_df.describe())
print("\nMissing values:", hr_analytics_df.isnull().sum())
print("\nData types:", hr_analytics_df.dtypes)
