import pandas as pd

df = pd.read_csv("dataset/fake_reviews_dataset.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel distribution:")
print(df["label"].value_counts(dropna=False))

print("\nCategory distribution:")
print(df["category"].value_counts())

print("Unique categories:", df["category"].nunique())

print("\nUnique ratings:")
print(sorted(df["rating"].unique()))

print("\nLabel values:")
print(df["label"].unique())

print("\nDuplicate records:", df.duplicated().sum())

print("\nAverage review length:")
print(df["text_"].str.len().describe())