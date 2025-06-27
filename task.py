import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data (ensure file is in the same folder or give correct path)
data = pd.read_csv("Crop_recommendation.csv")

# Print data overview
print(data.head())
print("Shape:", data.shape)
print("Missing values:\n", data.isnull().sum())

# Split features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
pickle.dump(model, open("model.pkl", "wb"))
# Evaluate
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
