
import joblib

# Load the saved preprocessor
preprocessor = joblib.load("models/preprocessor.pkl")

# Load the trained Logistic Regression model
model = joblib.load("models/best_model.pkl")

print("Fake Review Detection System")
print("Model and preprocessor loaded successfully!")




import pandas as pd

# Available product categories
categories = [
    "Kindle_Store_5",
    "Books_5",
    "Pet_Supplies_5",
    "Home_and_Kitchen_5",
    "Electronics_5",
    "Sports_and_Outdoors_5",
    "Tools_and_Home_Improvement_5",
    "Clothing_Shoes_and_Jewelry_5",
    "Toys_and_Games_5",
    "Movies_and_TV_5"
]

# Take review text
review = input("\nEnter product review: ").strip()

# Take rating
rating = float(input("Enter product rating (1-5): "))

# Take category
print("\nAvailable categories:")
for item in categories:
    print("-", item)

category = input("\nEnter product category: ").strip()

# Validate input
if not review:
    print("Review cannot be empty!")

elif rating < 1 or rating > 5:
    print("Rating must be between 1 and 5!")

elif category not in categories:
    print("Invalid product category!")

else:
    review_data = pd.DataFrame({
        "text_": [review],
        "rating": [rating],
        "category": [category]
    })

    print("\nReview details received successfully!")
    print(review_data)


# Transform user input using the saved preprocessor
review_transformed = preprocessor.transform(review_data)

# Generate prediction
prediction = model.predict(review_transformed)

print("\n" + "=" * 40)
print("         PREDICTION RESULT")
print("=" * 40)

if prediction[0] == 1:
    print("Result: FAKE REVIEW (CG)")
else:
    print("Result: GENUINE REVIEW (OR)")

print("=" * 40)