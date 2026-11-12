import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_reviews.csv")

# Separate input features and target labels
X = df[["text_", "rating", "category"]]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load fitted preprocessor
preprocessor = joblib.load("models/preprocessor.pkl")

# Transform training and testing data
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print("Training pipeline prepared successfully!")

print("Training shape:", X_train_transformed.shape)
print("Testing shape:", X_test_transformed.shape)



# Initialize Multinomial Naive Bayes
nb_model = MultinomialNB()

# Train the model
nb_model.fit(X_train_transformed, y_train)

print("\nMultinomial Naive Bayes trained successfully!")

# Check training and testing accuracy
print("Training accuracy:", nb_model.score(
    X_train_transformed, y_train
))

print("Testing accuracy:", nb_model.score(
    X_test_transformed, y_test
))



# Initialize Logistic Regression
lr_model = LogisticRegression(max_iter=1000)

# Train the model
lr_model.fit(X_train_transformed, y_train)

print("\nLogistic Regression trained successfully!")

# Check accuracy
print("Training accuracy:", lr_model.score(
    X_train_transformed, y_train
))

print("Testing accuracy:", lr_model.score(
    X_test_transformed, y_test
))


# Initialize Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train_transformed, y_train)

print("\nRandom Forest trained successfully!")

# Check accuracy
print("Training accuracy:", rf_model.score(
    X_train_transformed, y_train
))

print("Testing accuracy:", rf_model.score(
    X_test_transformed, y_test
))
