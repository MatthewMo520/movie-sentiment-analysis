import pandas as pd

df = pd.read_csv('IMDB Dataset.csv')

print(df.head())
print(df.info())
print(df['sentiment'].value_counts())