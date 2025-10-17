import pandas as pd

# Load the IMDB dataset
df = pd.read_csv('IMDB Dataset.csv')

print("Example reviews:")
print("positive review:")
print(df[df['sentiment'] == 'positive']['review'].iloc[0])

print("-"*50)

print("negative review:")
print(df[df['sentiment'] == 'negative']['review'].iloc[0])

df['review_length'] = df['review'].apply(len)
print("\nReview length statistics:")
print(f"average review length: {df['review_length'].mean():.0f} characters")
print(f"shortest review length: {df['review_length'].min()} characters")
print(f"longest review length: {df['review_length'].max()} characters")
