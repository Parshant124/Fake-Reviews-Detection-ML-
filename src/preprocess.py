import pandas as pd

# Load dataset
df = pd.read_csv("dataset/fake_reviews_dataset.csv")

print("Original shape:", df.shape)

# Display sample records
print(df.head())

# Check data types
print(df.dtypes)

# Check missing values
print(df.isnull().sum())

# Remove exact duplicate records
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

df["label"] = df["label"].map({
    "CG": 1,
    "OR": 0
})

# Remove leading and trailing spaces
df["text_"] = df["text_"].str.strip()

# Convert text to lowercase
df["text_"] = df["text_"].str.lower()

# Convert rating to numeric format
df["rating"] = pd.to_numeric(df["rating"])

# Save the cleaned dataset
df.to_csv("dataset/cleaned_reviews.csv", index=False)

print("\nPreprocessing completed!")
print("Final shape:", df.shape)
print(df.head())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nTotal labels:", df["label"].value_counts().sum())

print("\nMissing values:")
print(df.isnull().sum())