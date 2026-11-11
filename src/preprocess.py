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
df.drop_duplicates(inplace=True)

print("Shape after removing duplicates:", df.shape)