import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_reviews.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

from sklearn.model_selection import train_test_split

# Separate input features and target label
X = df[["text_", "rating", "category"]]
y = df["label"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDataset splitting completed!")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

# Define feature transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=5000), "text_"),
        ("rating", "passthrough", ["rating"]),
        ("category", OneHotEncoder(handle_unknown="ignore"), ["category"])
    ]
)

print("\nFeature transformers initialized successfully!")


# Fit transformers on training data and transform it
X_train_transformed = preprocessor.fit_transform(X_train)

# Transform testing data using the fitted transformers
X_test_transformed = preprocessor.transform(X_test)

print("\nFeature extraction completed!")

print("Training feature shape:", X_train_transformed.shape)
print("Testing feature shape:", X_test_transformed.shape)

# Save the fitted preprocessor
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("\nPreprocessor saved successfully!")